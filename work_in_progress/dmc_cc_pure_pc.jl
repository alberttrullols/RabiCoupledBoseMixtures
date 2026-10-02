include("core_cc.jl")
include("reblocking.jl")

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

# Pre-allocated workspace (Float64 spin)
mutable struct DMCWorkspace_2comp_pure
    walker_positions::Matrix{Float64} # (num_walkers, N_total)
    walker_spin::Matrix{Float64} # (num_walkers, N_total): 1.0 or 2.0
    new_positions::Matrix{Float64} # buffer for next generation
    new_spin::Matrix{Float64} # buffer for spin after branching
    M::Vector{Int} # copies per walker
    E_local::Vector{Float64} # local energy per walker
    new_E_local::Vector{Float64} # replicated local energies after branching

    # Pure Coordinates
    pure_positions::Matrix{Float64} # Snapshot of positions at block start
    pure_spin::Matrix{Float64}  # Snapshot of spins at block start
    new_pure_positions::Matrix{Float64} # Buffer for branching
    new_pure_spin::Matrix{Float64} # Buffer for branching
end

function DMCWorkspace_2comp_pure(num_walkers::Int, N_total::Int)
    max_walkers = num_walkers * MAX_COPIES  # worst case
    return DMCWorkspace_2comp_pure(
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Vector{Int}(undef, num_walkers),
        Vector{Float64}(undef, num_walkers),
        Vector{Float64}(undef, max_walkers),

        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total)
    )
end

function DMC_pure_pc(num_steps::Int, num_walkers::Int, dt::Float64, E_T_init::Float64,
    alpha11::Float64, alpha22::Float64, alpha12::Float64,
    a11::Float64, a22::Float64, a12::Float64,
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
    acum_mixed::Bool=true,
    equilibration_steps::Int=0,
)
    0 <= equilibration_steps < num_steps ||
        throw(ArgumentError("equilibration_steps must be between 0 and num_steps - 1"))
    N_total = N1 + N2
    n = 1.
    L = N_total/n
    invL = 1.0 / L

    #################################################################### parameter initialization
    t_start = time_ns()
    rng = MersenneTwister(seed)
    num_walkers_initial = num_walkers
    diffusion_scale = sqrt(2.0 * D_const * dt)
    E_T = E_T_init
    C_pop = 1.0 / (dt)    # population control constant
    energies = zeros(Float64, num_steps)
    energies_squared = zeros(Float64, num_steps)
    polarizations_P_mixed = zeros(Float64, num_steps)
    polarizations_P2_mixed = zeros(Float64, num_steps)
    polarizations_P4_mixed = zeros(Float64, num_steps)
    polarizations_P_abs_mixed = zeros(Float64, num_steps)

    equilibration_cutoff = equilibration_steps
    
    k_min = 2π / L
    Nk_vals = 75  # number of k values to compute
    Nbins_corr = 100
    dr = (L / 2) / Nbins_corr

    ws = DMCWorkspace_2comp_pure(num_walkers, N_total)

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
    n_buffers = Threads.nthreads() + 1  # +1 to cover potential interactive thread
    x_buffers             = [zeros(N_total) for _ in 1:n_buffers]
    drift_buffers         = [zeros(N_total) for _ in 1:n_buffers]
    drift_old_buffers     = [zeros(N_total) for _ in 1:n_buffers]
    rand_buffers          = [zeros(N_total) for _ in 1:n_buffers]
    spin_buffers          = [zeros(Float64, N_total) for _ in 1:n_buffers]
    logR_buffers          = [zeros(N_total) for _ in 1:n_buffers]
    rng_threads           = [MersenneTwister(seed + tid - 1) for tid in 1:n_buffers]
    
    # Thread-local structure factor and correlation buffers
    Sk_threads       = [zeros(Float64, Nk_vals) for _ in 1:n_buffers]
    Cs_threads       = [zeros(Float64, Nbins_corr) for _ in 1:n_buffers]
    Cs_count_threads = [zeros(Int, Nbins_corr) for _ in 1:n_buffers]
    P_bins_threads   = [zeros(Float64, N_total + 1) for _ in 1:n_buffers]

    block_size = forward_walk_steps
    pure_block_count = 0

    Sk_acc_pure       = zeros(Float64, Nk_vals)
    Cs                = zeros(Float64, Nbins_corr)
    Cs_count          = zeros(Int, Nbins_corr)
    P_bins            = zeros(Float64, N_total + 1)
    P_series_pure     = Float64[]
    P2_series_pure    = Float64[]
    P4_series_pure    = Float64[]
    P_abs_series_pure = Float64[]

    #################################################################### dmc loop
    for step in 1:num_steps

        if (step - 1) % block_size == 0
            if size(ws.pure_positions, 1) < num_walkers
                ws.pure_positions = Matrix{Float64}(undef, num_walkers, N_total)
                ws.pure_spin = Matrix{Float64}(undef, num_walkers, N_total)
            end

            # Copy current state to the historical snapshot
            @inbounds for w in 1:num_walkers
                for i in 1:N_total
                    ws.pure_positions[w, i] = ws.walker_positions[w, i]
                    ws.pure_spin[w, i] = ws.walker_spin[w, i]
                end
            end
        end

        if length(ws.M) < num_walkers
            resize!(ws.M, num_walkers)
            resize!(ws.E_local, num_walkers)
        end

        # walker propagation        
        Threads.@threads for w in 1:num_walkers
            tid = threadid()

            xb         = x_buffers[tid]
            F_old      = drift_old_buffers[tid]
            F_new      = drift_buffers[tid]
            randb      = rand_buffers[tid]
            spin_buf   = spin_buffers[tid]
            logR_acc   = logR_buffers[tid]
            rng_tid    = rng_threads[tid]

            # Load walker and spin
            @inbounds for i in 1:N_total
                xb[i] = ws.walker_positions[w, i]
                spin_buf[i] = ws.walker_spin[w, i]
            end
            
            # Compute drift and energy at old position (turbo SIMD pass)
            kin_old, tun_old, field_old = turbo_compute_drift_and_energy_piecewise!(F_old, logR_acc, xb, spin_buf,
                t, h; N_total, L, invL, alpha11, alpha22, alpha12, a11, a22, a12)
            E_old = kin_old + tun_old + field_old

            # Propagate walker
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
                    # Save F_old for predictor-corrector
                    @inbounds @simd for i in 1:N_total
                        F_new[i] = F_old[i]
                    end

                    # Predictor
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(xb[i] +
                                    2.0 * D_const * F_new[i] * dt +
                                    randb[i] * diffusion_scale, L)
                    end

                    # Compute drift at predictor position
                    turbo_compute_drift_and_energy_piecewise!(F_new, logR_acc, xb, spin_buf,
                        0.0, 0.0; N_total, L, invL, alpha11, alpha22, alpha12, a11, a22, a12)

                    # Corrector
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(ws.walker_positions[w, i] +
                                    D_const * (F_old[i] + F_new[i]) * dt +
                                    randb[i] * diffusion_scale, L)
                    end
                end
            else
                # Pure diffusion no drift
                @inbounds @simd for i in 1:N_total
                    xb[i] = mod(xb[i] +
                                randb[i] * diffusion_scale, L)
                end
            end

            # Compute drift and energy at new position
            kin_new, tun_new, field_new = turbo_compute_drift_and_energy_piecewise!(F_new, logR_acc, xb, spin_buf,
                t, h; N_total, L, invL, alpha11, alpha22, alpha12, a11, a22, a12)
            E_new = kin_new + tun_new + field_new

            # single-particle spin flips.
            # After each accepted flip, logR_acc is recomputed so the next particle's
            # flip probability reflects the updated spin configuration.
            if t != 0.0
                @inbounds for i in 1:N_total
                    if use_full_flip_expression
                        # Full expression with prefactor and exponent
                        logR = logR_acc[i]
                        R = exp(logR)
                        R2 = R^2
                        
                        # Prefactor: Psi_T(-s)^2 / (Psi_T(s)^2 + Psi_T(-s)^2)
                        prefactor = R2 / (1.0 + R2)
                        
                        # Exponent: -t * [R + 1/R] * dt
                        exponent = -t * ((R2 + 1.0) / R) * dt
                        
                        p_flip = prefactor * (1.0 - exp(exponent))
                    else
                        # Simple expression
                        p_flip = min(t * dt * exp(logR_acc[i]), 1.0)
                    end

                    if rand(rng_tid) < p_flip
                        spin_buf[i] = ifelse(spin_buf[i] == 1.0, 2.0, 1.0)
                        # Recompute logR_acc with the updated spin before the next flip
                        kin_new, tun_new, field_new = turbo_compute_drift_and_energy_piecewise!(F_new, logR_acc, xb, spin_buf,
                            t, h; N_total, L, invL, alpha11, alpha22, alpha12, a11, a22, a12)
                        E_new = kin_new + tun_new + field_new
                    end
                end
            end

            ws.E_local[w] = E_new

            # Branching
            if branching
                if isfinite(E_old) && isfinite(E_new)
                    if time_order == 2
                        wi = exp(-dt * (E_old+ E_new - 2.0 * E_T) / 2.0)
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

            # Store walker and spin
            @inbounds for i in 1:N_total
                ws.walker_positions[w, i] = xb[i]
                ws.walker_spin[w, i] = spin_buf[i]
            end
        end


        # Branching
        total_new = sum(ws.M[1:num_walkers])
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

        # Ensure new_pure arrays are sized correctly
        if size(ws.new_pure_positions, 1) < total_new
            ws.new_pure_positions = Matrix{Float64}(undef, total_new, N_total)
            ws.new_pure_spin = Matrix{Float64}(undef, total_new, N_total)
        end

        idx = 1
        @inbounds for w in 1:num_walkers
            for _ in 1:ws.M[w]
                for i in 1:N_total
                    ws.new_positions[idx, i] = ws.walker_positions[w, i]
                    ws.new_spin[idx, i] = ws.walker_spin[w, i]

                    # Replicate the pure coordinates
                    ws.new_pure_positions[idx, i] = ws.pure_positions[w, i]
                    ws.new_pure_spin[idx, i] = ws.pure_spin[w, i]
                end
                ws.new_E_local[idx] = ws.E_local[w]
                idx += 1
            end
        end

        num_walkers = total_new

        if size(ws.walker_positions, 1) < num_walkers
            ws.walker_positions = Matrix{Float64}(undef, num_walkers, N_total)
        end
        if size(ws.walker_spin, 1) < num_walkers
            ws.walker_spin = Matrix{Float64}(undef, num_walkers, N_total)
        end

        # Ensure main pure arrays are sized correctly
        if size(ws.pure_positions, 1) < num_walkers
            ws.pure_positions = Matrix{Float64}(undef, num_walkers, N_total)
            ws.pure_spin = Matrix{Float64}(undef, num_walkers, N_total)
        end

        @inbounds for w in 1:num_walkers
            for i in 1:N_total
                ws.walker_positions[w, i] = ws.new_positions[w, i]
                ws.walker_spin[w, i] = ws.new_spin[w, i]

                # Finalize the pure coordinates for the next step
                ws.pure_positions[w, i] = ws.new_pure_positions[w, i]
                ws.pure_spin[w, i] = ws.new_pure_spin[w, i]
            end
        end

        # Energy estimator
        if length(ws.E_local) < num_walkers
            resize!(ws.E_local, num_walkers)
        end
        E_sum = 0.0
        @inbounds for w in 1:num_walkers
            ws.E_local[w] = ws.new_E_local[w]
            E_sum += ws.E_local[w]
        end

        E_mean = E_sum / num_walkers
        E_var = var(ws.E_local[1:num_walkers])
        energies[step] = E_mean
        energies_squared[step] = E_mean^2

        if acum_mixed
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
        end

        # histo, Cs, and Sk calculations
        if step > equilibration_cutoff && step % block_size == 0

            # Pure polarization
            P_step_pure     = 0.0
            P2_step_pure    = 0.0
            P4_step_pure    = 0.0
            P_abs_step_pure = 0.0
            for tid in 1:num_threads_to_use
                fill!(P_bins_threads[tid], 0.0)
            end
            Threads.@threads for w in 1:num_walkers
                tid = Threads.threadid()
                n1 = 0
                for i in 1:N_total
                    n1 += (ws.pure_spin[w, i] == 1.0)
                end
                P_bins_threads[tid][n1 + 1] += 1.0
            end
            for tid in 1:num_threads_to_use
                P_bins .+= P_bins_threads[tid]
            end
            for w in 1:num_walkers
                n1 = 0
                for i in 1:N_total
                    # Use pure_spin
                    n1 += (ws.pure_spin[w, i] == 1.0)
                end
                P_w = (2*n1 - N_total) / N_total
                P_step_pure     += P_w
                P2_step_pure    += P_w^2
                P4_step_pure    += P_w^4
                P_abs_step_pure += abs(P_w)
            end
            push!(P_series_pure,     P_step_pure     / num_walkers)
            push!(P2_series_pure,    P2_step_pure    / num_walkers)
            push!(P4_series_pure,    P4_step_pure    / num_walkers)
            push!(P_abs_series_pure, P_abs_step_pure / num_walkers)

            # Pure Structure Factor and Spin Correlation
            for tid in 1:num_threads_to_use
                fill!(Sk_threads[tid], 0.0)
                fill!(Cs_threads[tid], 0.0)
                fill!(Cs_count_threads[tid], 0)
            end

            Threads.@threads for w in 1:num_walkers
                tid = Threads.threadid()
                # Sk
                for k_idx in 1:Nk_vals
                    k = (k_idx - 1) * k_min
                    re_spin = 0.0; im_spin = 0.0
                    for i in 1:N_total
                        s = (ws.pure_spin[w,i] == 1.0) ? 1.0 : -1.0
                        x = ws.pure_positions[w,i]
                        re_spin += s * cos(k * x)
                        im_spin += s * sin(k * x)
                    end
                    Sk_threads[tid][k_idx] += (re_spin^2 + im_spin^2) / (num_walkers * N_total)
                end
                # Cs(r)
                for i in 1:N_total-1
                    si = (ws.pure_spin[w,i] == 1.0) ? 1.0 : -1.0
                    xi = ws.pure_positions[w,i]
                    for j in i+1:N_total
                        sj = (ws.pure_spin[w,j] == 1.0) ? 1.0 : -1.0
                        xj = ws.pure_positions[w,j]
                        dx = xi - xj
                        dx -= L * round(dx * invL)
                        r = abs(dx)
                        if r < L/2
                            bin = Int(floor(r / dr)) + 1
                            if bin <= Nbins_corr
                                Cs_threads[tid][bin]       += si * sj / num_walkers
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

        # Population control
        if branching
            E_T = E_mean - C_pop *
                           log(num_walkers / num_walkers_initial)
        end

        if verbose && step % 100 == 0
            @printf("Step %d | walkers: %d | E_T = %.10f | E = %.10f | Var(E_loc) = %.10f\n",
                step, num_walkers, E_T, E_mean, E_var)
        end
    end

    # Calculate final pure estimators
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
        P2_avg_pure    = 0.0
        P4_avg_pure    = 0.0
        P_abs_avg_pure = 0.0
        P_err_pure     = 0.0
        P2_err_pure    = 0.0
        P4_err_pure    = 0.0
        P_abs_err_pure = 0.0
    end

    start_idx = equilibration_steps + 1
    energy_avg = mean(energies[start_idx:end])
    energy_squared_avg = mean(energies_squared[start_idx:end])
    if acum_mixed
        P_avg_mixed = mean(polarizations_P_mixed[start_idx:end])
        P2_avg_mixed = mean(polarizations_P2_mixed[start_idx:end])
        P4_avg_mixed = mean(polarizations_P4_mixed[start_idx:end])
        P_abs_avg_mixed = mean(polarizations_P_abs_mixed[start_idx:end])
    else
        P_avg_mixed = NaN
        P2_avg_mixed = NaN
        P4_avg_mixed = NaN
        P_abs_avg_mixed = NaN
    end

    ### Saving function files

    # Write polarization histogram
    open("hist.txt","w") do f
        println(f, "# P_value  count")
        for n1 in 0:N_total
            P_val = (2*n1 - N_total) / N_total
            println(f, P_val, " ", P_bins[n1 + 1])
        end
    end

    # Write Sk at all k values (spin structure factor)
    open("Sk.txt","w") do f
        println(f, "# k values and corresponding S(k) [pure estimator]")
        for k_idx in 1:Nk_vals
            k = (k_idx - 1) * k_min
            println(f, k, " ", Sk_avg_pure[k_idx])
        end
    end

    # Write Cs(r) (spin correlation function)
    open("Cs.txt","w") do f
        println(f, "# r  Cs(r) [pure estimator]")
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cs_full[i])
        end
    end

    # Write energy stream (one value per step, after equilibration)
    open("energy_stream.txt","w") do f
        println(f, "# step  E_mean")
        for i in start_idx:num_steps
            println(f, i, " ", energies[i])
        end
    end

    open("P_mixed_stream.txt", "w") do f
        println(f, acum_mixed ? "# step  <P>_mixed" : "# not accumulated (acum_mixed=false)")
        if acum_mixed
            for i in start_idx:num_steps
                println(f, i, " ", polarizations_P_mixed[i])
            end
        end
    end
    open("P2_mixed_stream.txt", "w") do f
        println(f, acum_mixed ? "# step  <P^2>_mixed" : "# not accumulated (acum_mixed=false)")
        if acum_mixed
            for i in start_idx:num_steps
                println(f, i, " ", polarizations_P2_mixed[i])
            end
        end
    end
    open("P4_mixed_stream.txt", "w") do f
        println(f, acum_mixed ? "# step  <P^4>_mixed" : "# not accumulated (acum_mixed=false)")
        if acum_mixed
            for i in start_idx:num_steps
                println(f, i, " ", polarizations_P4_mixed[i])
            end
        end
    end
    open("Pabs_mixed_stream.txt", "w") do f
        println(f, acum_mixed ? "# step  <|P|>_mixed" : "# not accumulated (acum_mixed=false)")
        if acum_mixed
            for i in start_idx:num_steps
                println(f, i, " ", polarizations_P_abs_mixed[i])
            end
        end
    end

    # Write polarization moment streams (one file per observable)
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
    open("P_stream.txt","w") do f
        println(f, "# block  <P>")
        for b in 1:length(P_series_pure)
            println(f, b, " ", P_series_pure[b])
        end
    end
    open("Pabs_stream.txt","w") do f
        println(f, "# block  <|P|>")
        for b in 1:length(P_abs_series_pure)
            println(f, b, " ", P_abs_series_pure[b])
        end
    end

    ################################################################### error estimation
    energy_err_unc = std(energies[start_idx:end]) / sqrt(length(energies[start_idx:end]))

    if reblocking
        opt_block_size, energy_err, _, _ = run_reblocking_loop(energies[start_idx:end], 40; label="energy")
        if acum_mixed
            _, P_err_mixed, _, _ = run_reblocking_loop(polarizations_P_mixed[start_idx:end], 40; label="P_mixed")
            _, P2_err_mixed, _, _ = run_reblocking_loop(polarizations_P2_mixed[start_idx:end], 40; label="P2_mixed")
            _, P4_err_mixed, _, _ = run_reblocking_loop(polarizations_P4_mixed[start_idx:end], 40; label="P4_mixed")
            _, P_abs_err_mixed, _, _ = run_reblocking_loop(polarizations_P_abs_mixed[start_idx:end], 40; label="Pabs_mixed")
        else
            P_err_mixed = NaN
            P2_err_mixed = NaN
            P4_err_mixed = NaN
            P_abs_err_mixed = NaN
        end
        if verbose
            println("Reblocking error estimation result:")
            println("  Optimal Block Size = ", opt_block_size)
            println("  Energy Error = ", energy_err)
        end
    else
        energy_err = energy_err_unc
        if acum_mixed
            sample_count = num_steps - equilibration_steps
            P_err_mixed = std(polarizations_P_mixed[start_idx:end]) / sqrt(sample_count)
            P2_err_mixed = std(polarizations_P2_mixed[start_idx:end]) / sqrt(sample_count)
            P4_err_mixed = std(polarizations_P4_mixed[start_idx:end]) / sqrt(sample_count)
            P_abs_err_mixed = std(polarizations_P_abs_mixed[start_idx:end]) / sqrt(sample_count)
        else
            P_err_mixed = NaN
            P2_err_mixed = NaN
            P4_err_mixed = NaN
            P_abs_err_mixed = NaN
        end
    end
    # Pure estimator errors: blocks are separated by forward_walk_steps steps,
    # which exceeds the correlation length by design → simple SEM is correct.
    if pure_block_count > 1
        P_err_pure     = std(P_series_pure)     / sqrt(pure_block_count)
        P2_err_pure    = std(P2_series_pure)    / sqrt(pure_block_count)
        P4_err_pure    = std(P4_series_pure)    / sqrt(pure_block_count)
        P_abs_err_pure = std(P_abs_series_pure) / sqrt(pure_block_count)
    end

    

    #################################################################### print stuff
    elapsed_s = (time_ns() - t_start) / 1e9
    if verbose
        final_var = var(ws.E_local[1:num_walkers])
        
        # Final state from last step (averaged over all walkers)
        println("\n========== Final State (Last Step) ==========")
        @printf("Step         = %d\n", num_steps)
        @printf("E_last      = %.15f\n", energies[num_steps])
        
        println("\n========== DMC Results (Two-Component) ==========")
        @printf("Threads      = %d\n", num_threads_to_use)
        @printf("E_DMC       = %.15f ± %.2e\n", energy_avg, energy_err)
        @printf("E_VMC       = %.15f\n", E_T_init)
        @printf("<P>         = %.10f ± %.2e\n", P_avg_pure, P_err_pure)
        @printf("<P²>        = %.10f ± %.2e\n", P2_avg_pure, P2_err_pure)
        @printf("<P⁴>        = %.10f ± %.2e\n", P4_avg_pure, P4_err_pure)
        @printf("<|P|>       = %.10f ± %.2e\n", P_abs_avg_pure, P_abs_err_pure)
        if acum_mixed
            @printf("Mixed <P>   = %.10f ± %.2e\n", P_avg_mixed, P_err_mixed)
            @printf("Mixed <P²>  = %.10f ± %.2e\n", P2_avg_mixed, P2_err_mixed)
            @printf("Mixed <P⁴>  = %.10f ± %.2e\n", P4_avg_mixed, P4_err_mixed)
            @printf("Mixed <|P|> = %.10f ± %.2e\n", P_abs_avg_mixed, P_abs_err_mixed)
        else
            println("Mixed polarization moments: not accumulated")
        end
        @printf("Final walkers: %d\n", num_walkers)
        @printf("Elapsed time: %.2f s (%.2f min)\n", elapsed_s, elapsed_s / 60)
        println("=================================================")
    end

    # Write final scalar results to file
    open("results.txt", "w") do f
        @printf(f, "acum_mixed   %s\n", acum_mixed)
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
        @printf(f, "elapsed_s    %.2f\n",   elapsed_s)
    end

    return energy_avg, energy_err, P_avg_pure, P_err_pure, P2_avg_pure, P2_err_pure, P4_avg_pure, P4_err_pure, P_abs_avg_pure, P_abs_err_pure, energy_squared_avg
end
