if !isdefined(@__MODULE__, :turbo_compute_drift_and_energy!)
    include("core_cc.jl")
end
if !isdefined(@__MODULE__, :run_reblocking_loop)
    include("reblocking.jl")
end

using Statistics, Printf
using Base.Threads

# Diffusion constant
if !@isdefined(D_const)
    const D_const = hbar^2 / (2.0 * m)
end

# Maximum number of copies per walker
if !@isdefined(MAX_COPIES)
    const MAX_COPIES = 1000
end

# Workspace without pure-coordinate fields (managed by the external circular buffer)
mutable struct DMCWorkspace_pc_ov
    walker_positions::Matrix{Float64} # (num_walkers, N_total)
    walker_spin::Matrix{Float64}      # (num_walkers, N_total): 1.0 or 2.0
    new_positions::Matrix{Float64}    # buffer for next generation
    new_spin::Matrix{Float64}
    M::Vector{Int}                    # copies per walker
    E_local::Vector{Float64}
    new_E_local::Vector{Float64}
end

function DMCWorkspace_pc_ov(num_walkers::Int, N_total::Int)
    max_walkers = num_walkers * MAX_COPIES
    return DMCWorkspace_pc_ov(
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Vector{Int}(undef, num_walkers),
        Vector{Float64}(undef, num_walkers),
        Vector{Float64}(undef, max_walkers),
    )
end

function DMC_pure_pc_overlapping(num_steps::Int, num_walkers::Int, dt::Float64, E_T_init::Float64,
    k11_val::Float64, delta11_val::Float64,
    k22_val::Float64, delta22_val::Float64,
    k12_val::Float64, delta12_val::Float64,
    t::Float64,
    N1::Int, N2::Int,
    h::Float64;
    seed::Int=1234,
    importance_sampling::Bool=true,
    branching::Bool=true,
    time_order::Int=1,
    verbose::Bool=true,
    reblocking::Bool=true,
    num_threads_to_use::Int=Threads.nthreads(),
    use_full_flip_expression::Bool=false,
    forward_walk_steps::Int=100,
    measurement_stride::Int=1_000,
    equilibration_steps::Int=0,
)
    forward_walk_steps > 0 || throw(ArgumentError("forward_walk_steps must be positive"))
    measurement_stride > 0 || throw(ArgumentError("measurement_stride must be positive"))
    0 <= equilibration_steps < num_steps ||
        throw(ArgumentError("equilibration_steps must be between 0 and num_steps - 1"))
    N_total = N1 + N2
    n = 1.
    L = N_total/n
    invL = 1.0 / L

    #################################################################### parameter initialization
    rng = MersenneTwister(seed)
    num_walkers_initial = num_walkers
    diffusion_scale = sqrt(2.0 * D_const * dt)
    E_T = E_T_init
    C_pop = 1.0 / (dt)
    energies = zeros(Float64, num_steps)
    energies_squared = zeros(Float64, num_steps)
    polarizations_P_mixed = zeros(Float64, num_steps)
    polarizations_P2_mixed = zeros(Float64, num_steps)
    polarizations_P4_mixed = zeros(Float64, num_steps)
    polarizations_P_abs_mixed = zeros(Float64, num_steps)

    equilibration_cutoff = equilibration_steps

    k_min = 2π / L
    Nk_vals = 75
    Nbins_corr = 100
    dr = (L / 2) / Nbins_corr

    ws = DMCWorkspace_pc_ov(num_walkers, N_total)

    #################################################################### walkers initialization
    @inbounds for w in 1:num_walkers
        for i in 1:N1
            ws.walker_positions[w, i] = rand(rng) * L
            ws.walker_spin[w, i] = 1.0
        end
        for i in 1:N2
            ws.walker_positions[w, N1 + i] = rand(rng) * L
            ws.walker_spin[w, N1 + i] = 2.0
        end
    end

    #################################################################### thread-local buffers
    num_threads_to_use = min(num_threads_to_use, Threads.nthreads())
    x_buffers             = [zeros(N_total) for _ in 1:num_threads_to_use]
    drift_buffers         = [zeros(N_total) for _ in 1:num_threads_to_use]
    drift_old_buffers     = [zeros(N_total) for _ in 1:num_threads_to_use]
    rand_buffers          = [zeros(N_total) for _ in 1:num_threads_to_use]
    spin_buffers          = [zeros(Float64, N_total) for _ in 1:num_threads_to_use]
    logR_buffers          = [zeros(N_total) for _ in 1:num_threads_to_use]
    rng_threads           = [MersenneTwister(seed + tid - 1) for tid in 1:num_threads_to_use]

    Sk_threads       = [zeros(Float64, Nk_vals) for _ in 1:num_threads_to_use]
    Cs_threads       = [zeros(Float64, Nbins_corr) for _ in 1:num_threads_to_use]
    Cs_count_threads = [zeros(Int, Nbins_corr) for _ in 1:num_threads_to_use]
    P_bins_threads   = [zeros(Float64, N_total + 1) for _ in 1:num_threads_to_use]
    P_threads        = zeros(Float64, num_threads_to_use)
    P2_threads       = zeros(Float64, num_threads_to_use)
    P4_threads       = zeros(Float64, num_threads_to_use)
    P_abs_threads    = zeros(Float64, num_threads_to_use)

    #################################################################### circular buffer for overlapping pure coordinates
    # Each slot j holds a snapshot replicated through branching until its forward lag
    # is reached, then the slot is measured and
    # the slot is reused for a fresh snapshot.
    #
    # Timing per step s (after equilibration):
    #   1. If buffer full: measure oldest slot (snap_write_idx) → push to series
    #   2. Save current walker state into snap_write_idx (overwriting the just-measured slot)
    #   3. Advance snap_write_idx (circular)
    #   4. Propagate walkers, then branch (replicate ALL active snapshot slots in sync)
    #
    # The lag is rounded up to the next measurement interval.

    window_size    = cld(forward_walk_steps, measurement_stride)
    verbose && println("Overlapping estimator stride=$measurement_stride, snapshot slots=$window_size")
    # Allocate conservatively; grow dynamically during branching if needed
    snap_pos    = [Matrix{Float64}(undef, max(num_walkers, 64), N_total) for _ in 1:window_size]
    snap_spin   = [Matrix{Float64}(undef, max(num_walkers, 64), N_total) for _ in 1:window_size]
    snap_nw     = zeros(Int, window_size)   # current row count (tracks num_walkers)
    snap_active = zeros(Bool, window_size)  # true once a slot has been written at least once
    snap_write_idx = 1   # next slot to overwrite
    buffer_filled  = 0   # number of slots currently active (0 → window_size)

    # Per-thread temporary buffers for parallel snapshot replication
    snap_tmp_pos_threads  = [Matrix{Float64}(undef, max(num_walkers, 64), N_total) for _ in 1:num_threads_to_use]
    snap_tmp_spin_threads = [Matrix{Float64}(undef, max(num_walkers, 64), N_total) for _ in 1:num_threads_to_use]

    #################################################################### accumulators
    Sk_acc_pure       = zeros(Float64, Nk_vals)
    Cs                = zeros(Float64, Nbins_corr)
    Cs_count          = zeros(Int, Nbins_corr)
    P_bins            = zeros(Float64, N_total + 1)
    P_series_pure     = Float64[]
    P2_series_pure    = Float64[]
    P4_series_pure    = Float64[]
    P_abs_series_pure = Float64[]
    pure_block_count  = 0

    #################################################################### dmc loop
    for step in 1:num_steps

        ################################################################ overlapping pure-coord management (before propagation)
        if step > equilibration_cutoff && step % measurement_stride == 0

            # --- Step 1: measure the oldest snapshot if buffer is full ---
            if buffer_filled == window_size
                j_meas  = snap_write_idx   # oldest slot
                nw_meas = snap_nw[j_meas]

                # Polarization moments
                P_step     = 0.0
                P2_step    = 0.0
                P4_step    = 0.0
                P_abs_step = 0.0
                for tid in 1:num_threads_to_use
                    fill!(P_bins_threads[tid], 0.0)
                    P_threads[tid]     = 0.0
                    P2_threads[tid]    = 0.0
                    P4_threads[tid]    = 0.0
                    P_abs_threads[tid] = 0.0
                end
                Threads.@threads for w in 1:nw_meas
                    tid = Threads.threadid()
                    n1 = 0
                    for i in 1:N_total
                        n1 += (snap_spin[j_meas][w, i] == 1.0)
                    end
                    P_w = (2*n1 - N_total) / N_total
                    P_bins_threads[tid][n1 + 1] += 1.0
                    P_threads[tid]     += P_w
                    P2_threads[tid]    += P_w^2
                    P4_threads[tid]    += P_w^4
                    P_abs_threads[tid] += abs(P_w)
                end
                for tid in 1:num_threads_to_use
                    P_bins     .+= P_bins_threads[tid]
                    P_step     += P_threads[tid]
                    P2_step    += P2_threads[tid]
                    P4_step    += P4_threads[tid]
                    P_abs_step += P_abs_threads[tid]
                end
                push!(P_series_pure,     P_step     / nw_meas)
                push!(P2_series_pure,    P2_step    / nw_meas)
                push!(P4_series_pure,    P4_step    / nw_meas)
                push!(P_abs_series_pure, P_abs_step / nw_meas)

                # Structure factor and spin correlation
                for tid in 1:num_threads_to_use
                    fill!(Sk_threads[tid], 0.0)
                    fill!(Cs_threads[tid], 0.0)
                    fill!(Cs_count_threads[tid], 0)
                end
                Threads.@threads for w in 1:nw_meas
                    tid = Threads.threadid()
                    for k_idx in 1:Nk_vals
                        k = (k_idx - 1) * k_min
                        re_spin = 0.0; im_spin = 0.0
                        for i in 1:N_total
                            s = (snap_spin[j_meas][w,i] == 1.0) ? 1.0 : -1.0
                            x = snap_pos[j_meas][w,i]
                            re_spin += s * cos(k * x)
                            im_spin += s * sin(k * x)
                        end
                        Sk_threads[tid][k_idx] += (re_spin^2 + im_spin^2) / (nw_meas * N_total)
                    end
                    for i in 1:N_total-1
                        si = (snap_spin[j_meas][w,i] == 1.0) ? 1.0 : -1.0
                        xi = snap_pos[j_meas][w,i]
                        for jj in i+1:N_total
                            sj = (snap_spin[j_meas][w,jj] == 1.0) ? 1.0 : -1.0
                            xj = snap_pos[j_meas][w,jj]
                            dx = xi - xj
                            dx -= L * round(dx * invL)
                            r  = abs(dx)
                            if r < L/2
                                bin = Int(floor(r / dr)) + 1
                                if bin <= Nbins_corr
                                    Cs_threads[tid][bin]       += si * sj / nw_meas
                                    Cs_count_threads[tid][bin] += 1
                                end
                            end
                        end
                    end
                end
                for tid in 1:num_threads_to_use
                    Sk_acc_pure .+= Sk_threads[tid]
                    Cs          .+= Cs_threads[tid]
                    Cs_count    .+= Cs_count_threads[tid]
                end
                pure_block_count += 1
            end

            # --- Step 2: save fresh snapshot into snap_write_idx (overwriting measured slot) ---
            j_write = snap_write_idx
            if size(snap_pos[j_write], 1) < num_walkers
                snap_pos[j_write]  = Matrix{Float64}(undef, num_walkers * 2, N_total)
                snap_spin[j_write] = Matrix{Float64}(undef, num_walkers * 2, N_total)
            end
            Threads.@threads for w in 1:num_walkers
                @inbounds for i in 1:N_total
                    snap_pos[j_write][w, i]  = ws.walker_positions[w, i]
                    snap_spin[j_write][w, i] = ws.walker_spin[w, i]
                end
            end
            snap_nw[j_write]     = num_walkers
            snap_active[j_write] = true

            # --- Step 3: advance circular index ---
            snap_write_idx = mod1(snap_write_idx + 1, window_size)
            if buffer_filled < window_size
                buffer_filled += 1
            end
        end

        ################################################################ resize M if needed
        if length(ws.M) < num_walkers
            resize!(ws.M, num_walkers)
            resize!(ws.E_local, num_walkers)
        end

        ################################################################ walker propagation
        Threads.@threads for w in 1:num_walkers
            tid = threadid()

            xb         = x_buffers[tid]
            F_old      = drift_old_buffers[tid]
            F_new      = drift_buffers[tid]
            randb      = rand_buffers[tid]
            spin_buf   = spin_buffers[tid]
            logR_acc   = logR_buffers[tid]
            rng_tid    = rng_threads[tid]

            @inbounds for i in 1:N_total
                xb[i] = ws.walker_positions[w, i]
                spin_buf[i] = ws.walker_spin[w, i]
            end

            kin_old, tun_old, field_old = turbo_compute_drift_and_energy!(F_old, logR_acc, xb, spin_buf,
                k11_val, delta11_val, k22_val, delta22_val, k12_val, delta12_val,
                t, h; N_total, L, invL)
            E_old = kin_old + tun_old + field_old

            randn!(rng_tid, randb)

            if importance_sampling
                if time_order == 1
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(
                            xb[i] +
                            2.0 * D_const * F_old[i] * dt +
                            randb[i] * diffusion_scale,
                            L
                        )
                    end

                elseif time_order == 2
                    @inbounds @simd for i in 1:N_total
                        F_new[i] = F_old[i]
                    end
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(xb[i] +
                                    2.0 * D_const * F_new[i] * dt +
                                    randb[i] * diffusion_scale, L)
                    end
                    turbo_compute_drift_and_energy!(F_new, logR_acc, xb, spin_buf,
                        k11_val, delta11_val, k22_val, delta22_val, k12_val, delta12_val,
                        0.0, 0.0; N_total, L, invL)
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(ws.walker_positions[w, i] +
                                    D_const * (F_old[i] + F_new[i]) * dt +
                                    randb[i] * diffusion_scale, L)
                    end
                end
            else
                @inbounds @simd for i in 1:N_total
                    xb[i] = mod(xb[i] + randb[i] * diffusion_scale, L)
                end
            end

            kin_new, tun_new, field_new = turbo_compute_drift_and_energy!(F_new, logR_acc, xb, spin_buf,
                k11_val, delta11_val, k22_val, delta22_val, k12_val, delta12_val,
                t, h; N_total, L, invL)
            E_new = kin_new + tun_new + field_new

            if t != 0.0
                @inbounds for i in 1:N_total
                    if use_full_flip_expression
                        logR = logR_acc[i]
                        R  = exp(logR)
                        R2 = R^2
                        prefactor = R2 / (1.0 + R2)
                        exponent  = -t * ((R2 + 1.0) / R) * dt
                        p_flip = prefactor * (1.0 - exp(exponent))
                    else
                        p_flip = min(t * dt * exp(logR_acc[i]), 1.0)
                    end

                    if rand(rng_tid) < p_flip
                        spin_buf[i] = ifelse(spin_buf[i] == 1.0, 2.0, 1.0)
                        kin_new, tun_new, field_new = turbo_compute_drift_and_energy!(F_new, logR_acc, xb, spin_buf,
                            k11_val, delta11_val, k22_val, delta22_val, k12_val, delta12_val,
                            t, h; N_total, L, invL)
                        E_new = kin_new + tun_new + field_new
                    end
                end
            end

            ws.E_local[w] = E_new

            if branching
                if isfinite(E_old) && isfinite(E_new)
                    if time_order == 2
                        wi = exp(-dt * (E_old + E_new - 2.0 * E_T) / 2.0)
                    elseif time_order == 1
                        wi = exp(-dt * (E_old - E_T))
                    end
                    wi = min(wi, Float64(MAX_COPIES))
                    ws.M[w] = min(floor(Int, wi + rand(rng_tid)), MAX_COPIES)
                    if ws.M[w] == MAX_COPIES
                        println("Warning: MAX_COPIES reached for walker $w at step $step")
                    end
                else
                    ws.M[w] = 0
                end
            else
                ws.M[w] = 1
            end

            @inbounds for i in 1:N_total
                ws.walker_positions[w, i] = xb[i]
                ws.walker_spin[w, i] = spin_buf[i]
            end
        end

        ################################################################ branching
        # Prefix sum for parallel scatter
        offsets = Vector{Int}(undef, num_walkers + 1)
        offsets[1] = 0
        @inbounds for w in 1:num_walkers
            offsets[w+1] = offsets[w] + ws.M[w]
        end
        total_new = offsets[num_walkers + 1]
        total_new == 0 && error("All walkers died.")

        if size(ws.new_positions, 1) < total_new
            ws.new_positions = Matrix{Float64}(undef, total_new, N_total)
        end
        if size(ws.new_spin, 1) < total_new
            ws.new_spin = Matrix{Float64}(undef, total_new, N_total)
        end
        if length(ws.new_E_local) < total_new
            resize!(ws.new_E_local, total_new)
        end

        # Replicate walkers in parallel
        Threads.@threads for w in 1:num_walkers
            base = offsets[w]
            @inbounds for c in 1:ws.M[w]
                idx = base + c
                for i in 1:N_total
                    ws.new_positions[idx, i] = ws.walker_positions[w, i]
                    ws.new_spin[idx, i]      = ws.walker_spin[w, i]
                end
                ws.new_E_local[idx] = ws.E_local[w]
            end
        end

        # Replicate every active pure snapshot in sync with walkers.
        # Parallelized over snapshot slots (each thread uses its own tmp buffer).
        Threads.@threads for j in 1:window_size
            snap_active[j] || continue
            tid = Threads.threadid()

            if size(snap_tmp_pos_threads[tid], 1) < total_new
                snap_tmp_pos_threads[tid]  = Matrix{Float64}(undef, total_new * 2, N_total)
                snap_tmp_spin_threads[tid] = Matrix{Float64}(undef, total_new * 2, N_total)
            end
            tmp_pos  = snap_tmp_pos_threads[tid]
            tmp_spin = snap_tmp_spin_threads[tid]

            @inbounds for w in 1:num_walkers
                base = offsets[w]
                for c in 1:ws.M[w]
                    idx2 = base + c
                    for i in 1:N_total
                        tmp_pos[idx2, i]  = snap_pos[j][w, i]
                        tmp_spin[idx2, i] = snap_spin[j][w, i]
                    end
                end
            end

            if size(snap_pos[j], 1) < total_new
                snap_pos[j]  = Matrix{Float64}(undef, total_new * 2, N_total)
                snap_spin[j] = Matrix{Float64}(undef, total_new * 2, N_total)
            end
            @inbounds for r in 1:total_new
                for i in 1:N_total
                    snap_pos[j][r, i]  = tmp_pos[r, i]
                    snap_spin[j][r, i] = tmp_spin[r, i]
                end
            end
            snap_nw[j] = total_new
        end

        num_walkers = total_new

        if size(ws.walker_positions, 1) < num_walkers
            ws.walker_positions = Matrix{Float64}(undef, num_walkers, N_total)
        end
        if size(ws.walker_spin, 1) < num_walkers
            ws.walker_spin = Matrix{Float64}(undef, num_walkers, N_total)
        end

        @inbounds for w in 1:num_walkers
            for i in 1:N_total
                ws.walker_positions[w, i] = ws.new_positions[w, i]
                ws.walker_spin[w, i]      = ws.new_spin[w, i]
            end
        end

        ################################################################ energy estimator
        if length(ws.E_local) < num_walkers
            resize!(ws.E_local, num_walkers)
        end
        E_sum = 0.0
        @inbounds for w in 1:num_walkers
            ws.E_local[w] = ws.new_E_local[w]
            E_sum += ws.E_local[w]
        end
        E_mean = E_sum / num_walkers
        E_var  = var(ws.E_local[1:num_walkers])
        energies[step]         = E_mean
        energies_squared[step] = E_mean^2

        P_mixed_sum = 0.0
        P2_mixed_sum = 0.0
        P4_mixed_sum = 0.0
        P_abs_mixed_sum = 0.0
        @inbounds for w in 1:num_walkers
            n1 = 0
            for i in 1:N_total
                n1 += (ws.walker_spin[w, i] == 1.0)
            end
            P_w = (2*n1 - N_total) / N_total
            P_mixed_sum += P_w
            P2_mixed_sum += P_w^2
            P4_mixed_sum += P_w^4
            P_abs_mixed_sum += abs(P_w)
        end
        polarizations_P_mixed[step] = P_mixed_sum / num_walkers
        polarizations_P2_mixed[step] = P2_mixed_sum / num_walkers
        polarizations_P4_mixed[step] = P4_mixed_sum / num_walkers
        polarizations_P_abs_mixed[step] = P_abs_mixed_sum / num_walkers

        ################################################################ population control
        if branching
            E_T = E_mean - C_pop * log(num_walkers / num_walkers_initial)
        end

        if verbose && step % 100 == 0
            @printf("Step %d | walkers: %d | E_T = %.10f | E = %.10f | Var(E_loc) = %.10f\n",
                step, num_walkers, E_T, E_mean, E_var)
        end
    end

    #################################################################### final pure estimators
    if pure_block_count > 0
        Sk_avg_pure    = Sk_acc_pure / pure_block_count
        Cs_full        = [Cs_count[i] > 0 ? Cs[i] / Cs_count[i] : 0.0 for i in 1:Nbins_corr]
        P_avg_pure     = mean(P_series_pure)
        P2_avg_pure    = mean(P2_series_pure)
        P4_avg_pure    = mean(P4_series_pure)
        P_abs_avg_pure = mean(P_abs_series_pure)
        P_err_pure     = std(P_series_pure)     / sqrt(pure_block_count)
        P2_err_pure    = std(P2_series_pure)    / sqrt(pure_block_count)
        P4_err_pure    = std(P4_series_pure)    / sqrt(pure_block_count)
        P_abs_err_pure = std(P_abs_series_pure) / sqrt(pure_block_count)
    else
        Sk_avg_pure    = zeros(Nk_vals)
        Cs_full        = zeros(Nbins_corr)
        P_avg_pure     = 0.0
        P2_avg_pure    = 0.0; P4_avg_pure    = 0.0; P_abs_avg_pure = 0.0
        P_err_pure     = 0.0
        P2_err_pure    = 0.0; P4_err_pure    = 0.0; P_abs_err_pure = 0.0
    end

    start_idx = equilibration_steps + 1
    energy_avg         = mean(energies[start_idx:end])
    energy_squared_avg = mean(energies_squared[start_idx:end])
    P_avg_mixed = mean(polarizations_P_mixed[start_idx:end])
    P2_avg_mixed = mean(polarizations_P2_mixed[start_idx:end])
    P4_avg_mixed = mean(polarizations_P4_mixed[start_idx:end])
    P_abs_avg_mixed = mean(polarizations_P_abs_mixed[start_idx:end])

    #################################################################### saving output files

    open("hist.txt","w") do f
        println(f, "# P_value  count")
        for n1 in 0:N_total
            P_val = (2*n1 - N_total) / N_total
            println(f, P_val, " ", P_bins[n1 + 1])
        end
    end

    open("Sk.txt","w") do f
        println(f, "# k values and corresponding S(k) [pure estimator, overlapping PC]")
        for k_idx in 1:Nk_vals
            k = (k_idx - 1) * k_min
            println(f, k, " ", Sk_avg_pure[k_idx])
        end
    end

    open("Cs.txt","w") do f
        println(f, "# r  Cs(r) [pure estimator, overlapping PC]")
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cs_full[i])
        end
    end

    open("energy_stream.txt","w") do f
        println(f, "# step  E_mean")
        for i in start_idx:num_steps
            println(f, i, " ", energies[i])
        end
    end

    open("P_mixed_stream.txt", "w") do f
        println(f, "# step  <P>_mixed")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P_mixed[i])
        end
    end
    open("P2_mixed_stream.txt", "w") do f
        println(f, "# step  <P^2>_mixed")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P2_mixed[i])
        end
    end
    open("P4_mixed_stream.txt", "w") do f
        println(f, "# step  <P^4>_mixed")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P4_mixed[i])
        end
    end
    open("Pabs_mixed_stream.txt", "w") do f
        println(f, "# step  <|P|>_mixed")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P_abs_mixed[i])
        end
    end

    open("P_stream.txt","w") do f
        println(f, "# block  <P>")
        for b in 1:length(P_series_pure)
            println(f, b, " ", P_series_pure[b])
        end
    end
    open("P2_stream.txt","w") do f
        println(f, "# block  <P^2>")
        for b in 1:length(P2_series_pure)
            println(f, b, " ", P2_series_pure[b])
        end
    end
    open("P4_stream.txt","w") do f
        println(f, "# block  <P^4>")
        for b in 1:length(P4_series_pure)
            println(f, b, " ", P4_series_pure[b])
        end
    end
    open("Pabs_stream.txt","w") do f
        println(f, "# block  <|P|>")
        for b in 1:length(P_abs_series_pure)
            println(f, b, " ", P_abs_series_pure[b])
        end
    end

    #################################################################### error estimation
    energy_err_unc = std(energies[start_idx:end]) / sqrt(length(energies[start_idx:end]))

    if reblocking
        opt_block_size, energy_err, _, _ = run_reblocking_loop(energies[start_idx:end], 40; label="energy")
        _, P_err_mixed, _, _ = run_reblocking_loop(polarizations_P_mixed[start_idx:end], 40; label="P_mixed")
        _, P2_err_mixed, _, _ = run_reblocking_loop(polarizations_P2_mixed[start_idx:end], 40; label="P2_mixed")
        _, P4_err_mixed, _, _ = run_reblocking_loop(polarizations_P4_mixed[start_idx:end], 40; label="P4_mixed")
        _, P_abs_err_mixed, _, _ = run_reblocking_loop(polarizations_P_abs_mixed[start_idx:end], 40; label="Pabs_mixed")
        if pure_block_count > 0
            _, P_err_pure, _, _     = run_reblocking_loop(P_series_pure, 40;     label="P")
            _, P2_err_pure, _, _    = run_reblocking_loop(P2_series_pure, 40;    label="P^2")
            _, P4_err_pure, _, _    = run_reblocking_loop(P4_series_pure, 40;    label="P^4")
            _, P_abs_err_pure, _, _ = run_reblocking_loop(P_abs_series_pure, 40; label="|P|")
        end
        if verbose
            println("Reblocking error estimation result:")
            println("  Optimal Block Size = ", opt_block_size)
            println("  Energy Error = ", energy_err)
        end
    else
        energy_err = energy_err_unc
        sample_count = num_steps - equilibration_steps
        P_err_mixed = std(polarizations_P_mixed[start_idx:end]) / sqrt(sample_count)
        P2_err_mixed = std(polarizations_P2_mixed[start_idx:end]) / sqrt(sample_count)
        P4_err_mixed = std(polarizations_P4_mixed[start_idx:end]) / sqrt(sample_count)
        P_abs_err_mixed = std(polarizations_P_abs_mixed[start_idx:end]) / sqrt(sample_count)
    end

    #################################################################### print results
    if verbose
        println("\n========== Final State (Last Step) ==========")
        @printf("Step         = %d\n", num_steps)
        @printf("E_last      = %.15f\n", energies[num_steps])

        println("\n========== DMC Results (overlapping pure-coordinates) ==========")
        @printf("Threads      = %d\n", num_threads_to_use)
        @printf("E_DMC       = %.15f ± %.2e\n", energy_avg, energy_err)
        @printf("E_VMC       = %.15f\n", E_T_init)
        @printf("<P>         = %.10f ± %.2e\n", P_avg_pure, P_err_pure)
        @printf("<P²>        = %.10f ± %.2e\n", P2_avg_pure, P2_err_pure)
        @printf("<P⁴>        = %.10f ± %.2e\n", P4_avg_pure, P4_err_pure)
        @printf("<|P|>       = %.10f ± %.2e\n", P_abs_avg_pure, P_abs_err_pure)
        @printf("Mixed <P>   = %.10f ± %.2e\n", P_avg_mixed, P_err_mixed)
        @printf("Mixed <P²>  = %.10f ± %.2e\n", P2_avg_mixed, P2_err_mixed)
        @printf("Mixed <P⁴>  = %.10f ± %.2e\n", P4_avg_mixed, P4_err_mixed)
        @printf("Mixed <|P|> = %.10f ± %.2e\n", P_abs_avg_mixed, P_abs_err_mixed)
        @printf("Final walkers: %d\n", num_walkers)
        println("=================================================================")
    end

    open("results.txt", "w") do f
        @printf(f, "E_DMC        %.15f\n", energy_avg)
        @printf(f, "E_DMC_err    %.6e\n",  energy_err)
        @printf(f, "E_sq_avg     %.15f\n", energy_squared_avg)
        @printf(f, "P_avg        %.10f\n", P_avg_pure)
        @printf(f, "P_err        %.6e\n",  P_err_pure)
        @printf(f, "P2_avg       %.10f\n", P2_avg_pure)
        @printf(f, "P2_err       %.6e\n",  P2_err_pure)
        @printf(f, "P4_avg       %.10f\n", P4_avg_pure)
        @printf(f, "P4_err       %.6e\n",  P4_err_pure)
        @printf(f, "Pabs_avg     %.10f\n", P_abs_avg_pure)
        @printf(f, "Pabs_err     %.6e\n",  P_abs_err_pure)
        @printf(f, "P_mixed_avg  %.10f\n", P_avg_mixed)
        @printf(f, "P_mixed_err  %.6e\n",  P_err_mixed)
        @printf(f, "P2_mixed_avg %.10f\n", P2_avg_mixed)
        @printf(f, "P2_mixed_err %.6e\n",  P2_err_mixed)
        @printf(f, "P4_mixed_avg %.10f\n", P4_avg_mixed)
        @printf(f, "P4_mixed_err %.6e\n",  P4_err_mixed)
        @printf(f, "Pabs_mixed_avg %.10f\n", P_abs_avg_mixed)
        @printf(f, "Pabs_mixed_err %.6e\n",  P_abs_err_mixed)
    end

    return energy_avg, energy_err, P_avg_pure, P_err_pure, P2_avg_pure, P2_err_pure, P4_avg_pure, P4_err_pure, P_abs_avg_pure, P_abs_err_pure, energy_squared_avg
end
