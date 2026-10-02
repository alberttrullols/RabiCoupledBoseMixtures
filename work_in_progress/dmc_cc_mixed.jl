if !isdefined(@__MODULE__, :turbo_compute_drift_and_energy_piecewise!)
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

# Pre-allocated workspace
mutable struct DMCWorkspace_2comp
    walker_positions::Matrix{Float64}   # (num_walkers, N_total)
    walker_spin::Matrix{Float64}        # (num_walkers, N_total): 1.0 or 2.0
    new_positions::Matrix{Float64}      # buffer for next generation
    new_spin::Matrix{Float64}           # buffer for spin after branching
    M::Vector{Int}                      # copies per walker
    E_local::Vector{Float64}            # local energy per walker
    new_E_local::Vector{Float64}        # replicated local energies after branching
end

function DMCWorkspace_2comp(num_walkers::Int, N_total::Int)
    max_walkers = num_walkers * MAX_COPIES
    return DMCWorkspace_2comp(
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Vector{Int}(undef, num_walkers),
        Vector{Float64}(undef, num_walkers),
        Vector{Float64}(undef, max_walkers)
    )
end

"""
Run Diffusion Monte Carlo for 1D two-component bosons with contact interactions.
Multi-threaded version with @turbo SIMD-vectorized physics and sequential single-particle spin flips.
Returns mixed estimators and writes flat-file output.
"""
function DMC_mixed(num_steps::Int, num_walkers::Int, dt::Float64, E_T_init::Float64,
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
    equilibration_steps::Int=0,
)
    0 <= equilibration_steps < num_steps ||
        throw(ArgumentError("equilibration_steps must be between 0 and num_steps - 1"))
    N_total = N1 + N2
    n = 1.
    L = N_total / n
    invL = 1.0 / L

    #################################################################### parameter initialization
    t_start = time_ns()
    rng = MersenneTwister(seed)
    num_walkers_initial = num_walkers
    diffusion_scale = sqrt(2.0 * D_const * dt)
    E_T = E_T_init
    C_pop = 1.0 / dt
    energies = zeros(Float64, num_steps)
    energies_squared = zeros(Float64, num_steps)
    polarizations_P = zeros(Float64, num_steps)
    polarizations_P2 = zeros(Float64, num_steps)
    polarizations_P4 = zeros(Float64, num_steps)
    polarizations_P_abs = zeros(Float64, num_steps)

    equilibration_cutoff = equilibration_steps
    histogram_stride = 100
    k_min = 2π / L
    Nk_vals = 75
    Sk_acc = zeros(Float64, Nk_vals)
    Sk_count = 0
    Sn_acc = zeros(Float64, Nk_vals)
    Sn_count = 0

    ws = DMCWorkspace_2comp(num_walkers, N_total)

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

    #################################################################### dmc loop
    for step in 1:num_steps
        if length(ws.M) < num_walkers
            resize!(ws.M, num_walkers)
            resize!(ws.E_local, num_walkers)
        end

        # walker propagation
        Threads.@threads for w in 1:num_walkers
            tid = threadid()

            xb       = x_buffers[tid]
            F_old    = drift_old_buffers[tid]
            F_new    = drift_buffers[tid]
            randb    = rand_buffers[tid]
            spin_buf = spin_buffers[tid]
            logR_acc = logR_buffers[tid]
            rng_tid  = rng_threads[tid]

            # Load walker and spin
            @inbounds for i in 1:N_total
                xb[i]       = ws.walker_positions[w, i]
                spin_buf[i] = ws.walker_spin[w, i]
            end

            # Compute drift and energy at old position
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
                # Pure diffusion, no drift
                @inbounds @simd for i in 1:N_total
                    xb[i] = mod(xb[i] + randb[i] * diffusion_scale, L)
                end
            end

            # Compute drift and energy at new position
            kin_new, tun_new, field_new = turbo_compute_drift_and_energy_piecewise!(F_new, logR_acc, xb, spin_buf,
                t, h; N_total, L, invL, alpha11, alpha22, alpha12, a11, a22, a12)
            E_new = kin_new + tun_new + field_new

            # Sequential single-particle spin flips.
            # After each accepted flip, logR_acc is recomputed so the next particle's
            # flip probability reflects the updated spin configuration.
            if t != 0.0
                @inbounds for i in 1:N_total
                    if use_full_flip_expression
                        logR = logR_acc[i]
                        R    = exp(logR)
                        R2   = R^2
                        prefactor = R2 / (1.0 + R2)
                        exponent  = -t * ((R2 + 1.0) / R) * dt
                        p_flip    = prefactor * (1.0 - exp(exponent))
                    else
                        p_flip = min(t * dt * exp(logR_acc[i]), 1.0)
                    end

                    if rand(rng_tid) < p_flip
                        spin_buf[i] = ifelse(spin_buf[i] == 1.0, 2.0, 1.0)
                        # Recompute logR_acc with updated spin before next flip
                        kin_new, tun_new, field_new = turbo_compute_drift_and_energy_piecewise!(F_new, logR_acc, xb, spin_buf,
                            t, h; N_total, L, invL, alpha11, alpha22, alpha12, a11, a22, a12)
                        E_new = kin_new + tun_new + field_new
                    end
                end
            end

            ws.E_local[w] = E_new

            # Branching weight
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

            # Store walker and spin
            @inbounds for i in 1:N_total
                ws.walker_positions[w, i] = xb[i]
                ws.walker_spin[w, i]      = spin_buf[i]
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

        idx = 1
        @inbounds for w in 1:num_walkers
            for _ in 1:ws.M[w]
                for i in 1:N_total
                    ws.new_positions[idx, i] = ws.walker_positions[w, i]
                    ws.new_spin[idx, i]      = ws.walker_spin[w, i]
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

        @inbounds for w in 1:num_walkers
            for i in 1:N_total
                ws.walker_positions[w, i] = ws.new_positions[w, i]
                ws.walker_spin[w, i]      = ws.new_spin[w, i]
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
        E_var  = var(ws.E_local[1:num_walkers])
        energies[step]         = E_mean
        energies_squared[step] = E_mean^2

        # Mixed polarization estimators (measured on current walker_spin)
        P_sum = 0.0; P2_sum = 0.0; P4_sum = 0.0; P_abs_sum = 0.0
        @inbounds for w in 1:num_walkers
            n1 = 0
            for i in 1:N_total
                n1 += (ws.walker_spin[w, i] == 1.0)
            end
            P_w = (2*n1 - N_total) / N_total
            P_sum     += P_w
            P2_sum    += P_w^2
            P4_sum    += P_w^4
            P_abs_sum += abs(P_w)
        end
        polarizations_P[step]     = P_sum     / num_walkers
        polarizations_P2[step]    = P2_sum    / num_walkers
        polarizations_P4[step]    = P4_sum    / num_walkers
        polarizations_P_abs[step] = P_abs_sum / num_walkers

        # Spin and density structure factors (sampled after equilibration)
        if step > equilibration_cutoff && step % histogram_stride == 0
            for k_idx in 1:Nk_vals
                k = (k_idx - 1) * k_min
                Sk_step = 0.0
                Sn_step = 0.0
                for w in 1:num_walkers
                    re_spin = 0.0; im_spin = 0.0
                    re_dens = 0.0; im_dens = 0.0
                    for i in 1:N_total
                        s = (ws.walker_spin[w, i] == 1.0) ? 1.0 : -1.0
                        x = ws.walker_positions[w, i]
                        re_spin += s * cos(k * x)
                        im_spin += s * sin(k * x)
                        re_dens += cos(k * x)
                        im_dens += sin(k * x)
                    end
                    Sk_step += (re_spin^2 + im_spin^2)
                    Sn_step += (re_dens^2 + im_dens^2)
                end
                Sk_acc[k_idx] += Sk_step / (num_walkers * N_total)
                Sn_acc[k_idx] += Sn_step / (num_walkers * N_total)
            end
            Sk_count += 1
            Sn_count += 1
        end

        # Population control
        if branching
            E_T = E_mean - C_pop * log(num_walkers / num_walkers_initial)
        end

        if verbose && step % 100 == 0
            @printf("Step %d | walkers: %d | E_T = %.10f | E = %.10f | Var(E_loc) = %.10f\n",
                step, num_walkers, E_T, E_mean, E_var)
        end
    end

    start_idx          = equilibration_steps + 1
    energy_avg         = mean(energies[start_idx:end])
    P_avg              = mean(polarizations_P[start_idx:end])
    P2_avg             = mean(polarizations_P2[start_idx:end])
    P4_avg             = mean(polarizations_P4[start_idx:end])
    P_abs_avg          = mean(polarizations_P_abs[start_idx:end])
    energy_squared_avg = mean(energies_squared[start_idx:end])

    ### Flat file output #####################################################

    # Energy stream (after equilibration)
    open("energy_stream.txt", "w") do f
        println(f, "# step  E_mean")
        for i in start_idx:num_steps
            println(f, i, " ", energies[i])
        end
    end

    # Polarization moment streams (after equilibration)
    open("P_stream.txt", "w") do f
        println(f, "# step  <P>")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P[i])
        end
    end
    open("P2_stream.txt", "w") do f
        println(f, "# step  <P^2>")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P2[i])
        end
    end
    open("P4_stream.txt", "w") do f
        println(f, "# step  <P^4>")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P4[i])
        end
    end
    open("Pabs_stream.txt", "w") do f
        println(f, "# step  <|P|>")
        for i in start_idx:num_steps
            println(f, i, " ", polarizations_P_abs[i])
        end
    end

    Sk_avg = Sk_count > 0 ? Sk_acc / Sk_count : zeros(Nk_vals)
    open("Sk.txt", "w") do f
        println(f, "# k  S_spin(k)")
        for k_idx in 1:Nk_vals
            println(f, (k_idx - 1) * k_min, " ", Sk_avg[k_idx])
        end
    end

    Sn_avg = Sn_count > 0 ? Sn_acc / Sn_count : zeros(Nk_vals)
    open("Sn.txt", "w") do f
        println(f, "# k  S_density(k)")
        for k_idx in 1:Nk_vals
            println(f, (k_idx - 1) * k_min, " ", Sn_avg[k_idx])
        end
    end

    ################################################################### error estimation
    energy_err_unc = std(energies[start_idx:end]) / sqrt(length(energies[start_idx:end]))

    if reblocking
        opt_block_size, energy_err, _, _ = run_reblocking_loop(energies[start_idx:end], 40; label="energy")
        _, P_err,     _, _ = run_reblocking_loop(polarizations_P[start_idx:end],     40)
        _, P2_err,    _, _ = run_reblocking_loop(polarizations_P2[start_idx:end],    40)
        _, P4_err,    _, _ = run_reblocking_loop(polarizations_P4[start_idx:end],    40)
        _, P_abs_err, _, _ = run_reblocking_loop(polarizations_P_abs[start_idx:end], 40)
        if verbose
            println("Reblocking error estimation result:")
            println("  Optimal Block Size = ", opt_block_size)
            println("  Energy Error = ", energy_err)
            println("  <P> Error = ", P_err)
            println("  <P²> Error = ", P2_err)
            println("  <P⁴> Error = ", P4_err)
            println("  <|P|> Error = ", P_abs_err)
        end
    else
        energy_err = energy_err_unc
        P_err      = std(polarizations_P[start_idx:end])     / sqrt(length(polarizations_P[start_idx:end]))
        P2_err     = std(polarizations_P2[start_idx:end])    / sqrt(length(polarizations_P2[start_idx:end]))
        P4_err     = std(polarizations_P4[start_idx:end])    / sqrt(length(polarizations_P4[start_idx:end]))
        P_abs_err  = std(polarizations_P_abs[start_idx:end]) / sqrt(length(polarizations_P_abs[start_idx:end]))
    end

    elapsed_s = (time_ns() - t_start) / 1e9

    #################################################################### print results
    if verbose
        println("\n========== Final State (Last Step) ==========")
        @printf("Step         = %d\n", num_steps)
        @printf("E_last      = %.15f\n", energies[num_steps])
        @printf("P²_last     = %.10f\n", polarizations_P2[num_steps])

        println("\n========== DMC Results (Two-Component, Mixed) ==========")
        @printf("Threads      = %d\n", num_threads_to_use)
        @printf("E_DMC       = %.15f ± %.2e\n", energy_avg, energy_err)
        @printf("E_VMC       = %.15f\n", E_T_init)
        @printf("<P>         = %.10f ± %.2e\n", P_avg, P_err)
        @printf("<P²>        = %.10f ± %.2e\n", P2_avg, P2_err)
        @printf("<P⁴>        = %.10f ± %.2e\n", P4_avg, P4_err)
        @printf("<|P|>       = %.10f ± %.2e\n", P_abs_avg, P_abs_err)
        @printf("Final walkers: %d\n", num_walkers)
        @printf("Elapsed time: %.2f s (%.2f min)\n", elapsed_s, elapsed_s / 60)
        println("=================================================")
    end

    # Write scalar results
    open("results.txt", "w") do f
        @printf(f, "E_DMC        %.15f\n", energy_avg)
        @printf(f, "E_DMC_err    %.6e\n",  energy_err)
        @printf(f, "E_sq_avg     %.15f\n", energy_squared_avg)
        @printf(f, "P_avg        %.10f\n", P_avg)
        @printf(f, "P_err        %.6e\n",  P_err)
        @printf(f, "P2_avg       %.10f\n", P2_avg)
        @printf(f, "P2_err       %.6e\n",  P2_err)
        @printf(f, "P4_avg       %.10f\n", P4_avg)
        @printf(f, "P4_err       %.6e\n",  P4_err)
        @printf(f, "Pabs_avg     %.10f\n", P_abs_avg)
        @printf(f, "Pabs_err     %.6e\n",  P_abs_err)
        @printf(f, "elapsed_s    %.2f\n",   elapsed_s)
    end

    return energy_avg, energy_err, P_avg, P_err, P2_avg, P2_err, P4_avg, P4_err, P_abs_avg, P_abs_err, energy_squared_avg
end
