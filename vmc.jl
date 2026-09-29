if !isdefined(@__MODULE__, :log_psiT)
    include("core_cc.jl")
end
using Printf

# MC parameters
const VMC_EQUILIBRATION_STEPS = 20_000
const VMC_PRODUCTION_STEPS = 80_000
const step_size = 0.4

"""
Standard Metropolis VMC with fixed N1 and N2 particles.
Only particle positions are updated, spin labels are frozen throughout the run.
"""
function run_metropolis(k11::Float64, delta11::Float64,
                       k22::Float64, delta22::Float64,
                       k12::Float64, delta12::Float64,
                       t::Float64,
                       h::Float64=0.0;
                       N1::Int, N2::Int, seed::Int=1234, verbose::Bool=true,
                       equilibration_steps::Int=VMC_EQUILIBRATION_STEPS,
                       production_steps::Int=VMC_PRODUCTION_STEPS)
    equilibration_steps >= 0 || throw(ArgumentError("equilibration_steps must be nonnegative"))
    production_steps > 0 || throw(ArgumentError("production_steps must be positive"))
    total_steps = equilibration_steps + production_steps
    N_total = N1 + N2
    n = 1.0
    L = N_total / n
    invL = 1.0 / L

    rng = MersenneTwister(seed)

    # Initialize positions and spins (Float64)
    x = rand(rng, N_total) .* L
    spin = zeros(Float64, N_total)
    for i in 1:N1
        spin[i] = 1.0
    end
    for i in N1+1:N_total
        spin[i] = 2.0
    end

    # Pre-allocate buffers
    F = zeros(N_total)
    logR_acc = zeros(N_total)

    logpsi = log_psiT(x, spin, k11, delta11, k22, delta22, k12, delta12; N_total, L, invL)

    E_lap_acc = 0.0
    E_lap_acc2 = 0.0
    nmeas = 0

    Nbins_corr = 100
    dr = (L/2) / Nbins_corr
    k_min = 2π / L
    Nk_vals = 100
    Cs      = zeros(Float64, Nbins_corr)
    Cs_count = zeros(Int, Nbins_corr)
    Cn      = zeros(Float64, Nbins_corr)
    Sk_acc  = zeros(Float64, Nk_vals)
    Sn_acc  = zeros(Float64, Nk_vals)
    Sk_count = 0

    for step in 1:total_steps
        if step % 10000 == 0
            println("Step ", step, "/", total_steps)
        end

        # Pick random particle and attempt move
        i = rand(rng, 1:N_total)
        x_new = copy(x)
        x_new[i] = pbc(x_new[i] + step_size * (rand(rng) - 0.5), L, invL)

        logpsi_new = log_psiT(x_new, spin, k11, delta11, k22, delta22, k12, delta12; N_total, L, invL)
        acc = min(1.0, exp(2.0 * (logpsi_new - logpsi)))

        if rand(rng) < acc
            x = x_new
            logpsi = logpsi_new
        end

        # Measurements after equilibration
        if step > equilibration_steps
            kinetic, tunnel, field = turbo_compute_drift_and_energy!(F, logR_acc, x, spin,
                        k11, delta11, k22, delta22, k12, delta12, t, h;
                        N_total, L, invL)
            E_lap = kinetic + tunnel + field
            E_lap_acc += E_lap
            E_lap_acc2 += E_lap^2
            nmeas += 1

            # Cs(r) and Cn(r)
            for i in 1:N_total-1
                si = (spin[i] == 1.0) ? 1.0 : -1.0
                xi = x[i]
                for j in i+1:N_total
                    sj = (spin[j] == 1.0) ? 1.0 : -1.0
                    xj = x[j]
                    dx = xi - xj
                    dx -= L * round(dx * invL)
                    r = abs(dx)
                    if r < L/2
                        bin = Int(floor(r / dr)) + 1
                        if bin <= Nbins_corr
                            Cs[bin]       += si * sj
                            Cs_count[bin] += 1
                            Cn[bin]       += 1.0
                        end
                    end
                end
            end

            # Sk(k) and Sn(k)
            for k_idx in 1:Nk_vals
                k_val = (k_idx - 1) * k_min
                re_spin = 0.0; im_spin = 0.0
                re_dens = 0.0; im_dens = 0.0
                for i in 1:N_total
                    s  = (spin[i] == 1.0) ? 1.0 : -1.0
                    xi = x[i]
                    re_spin += s * cos(k_val * xi)
                    im_spin += s * sin(k_val * xi)
                    re_dens += cos(k_val * xi)
                    im_dens += sin(k_val * xi)
                end
                Sk_acc[k_idx] += (re_spin^2 + im_spin^2) / N_total
                Sn_acc[k_idx] += (re_dens^2 + im_dens^2) / N_total
            end
            Sk_count += 1
        end
    end

    E_lap_mean = E_lap_acc / nmeas
    E_lap_var = E_lap_acc2 / nmeas - E_lap_mean^2
    E_lap_var = max(0.0, E_lap_var)
    E_lap_err = sqrt(E_lap_var / nmeas)

    mkpath("VMC")
    open("VMC/results.txt", "w") do f
        @printf(f, "N1                  %d\n", N1)
        @printf(f, "N2                  %d\n", N2)
        @printf(f, "E_VMC               %.15f\n", E_lap_mean)
        @printf(f, "E_VMC_err           %.6e\n", E_lap_err)
        @printf(f, "E_VMC_per_particle  %.15f\n", E_lap_mean / N_total)
        @printf(f, "equilibration_steps %d\n", equilibration_steps)
        @printf(f, "production_steps    %d\n", production_steps)
    end

    # Normalize and save correlators / structure factors
    Cs_full = zeros(Float64, Nbins_corr)
    for i in 1:Nbins_corr
        if Cs_count[i] > 0
            Cs_full[i] = Cs[i] / Cs_count[i]
        end
    end
    expected_total_pairs = nmeas * (N_total * (N_total - 1) / 2.0) * (dr / (L/2))
    Cn_full = Cn ./ expected_total_pairs
    Sk_avg  = Sk_count > 0 ? Sk_acc ./ Sk_count : zeros(Float64, Nk_vals)
    Sn_avg  = Sk_count > 0 ? Sn_acc ./ Sk_count : zeros(Float64, Nk_vals)

    base = @sprintf("N1%d_N2%d_t%.4f_h%.4f_vmc", N1, N2, t, h)
    N_folder = "N$N_total"
    mkpath("VMC/correlators/$N_folder")
    mkpath("VMC/spin_structure_factor/$N_folder")
    mkpath("VMC/density_structure_factor/$N_folder")
    open("VMC/correlators/$N_folder/Cs_full_$base.txt", "w") do f
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cs_full[i])
        end
    end
    open("VMC/correlators/$N_folder/Cn_full_$base.txt", "w") do f
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cn_full[i])
        end
    end
    open("VMC/spin_structure_factor/$N_folder/Sk_$base.txt", "w") do f
        println(f, "# k  S_s(k)")
        for k_idx in 1:Nk_vals
            k_val = (k_idx - 1) * k_min
            println(f, k_val, " ", Sk_avg[k_idx])
        end
    end
    open("VMC/density_structure_factor/$N_folder/Sn_$base.txt", "w") do f
        println(f, "# k  S_n(k)")
        for k_idx in 1:Nk_vals
            k_val = (k_idx - 1) * k_min
            println(f, k_val, " ", Sn_avg[k_idx])
        end
    end

    if verbose
        println("VMC result (unified spin-based):")
        println("Energy (via Laplacian) = ", E_lap_mean/L, " ± ", E_lap_err/L)
        println("Measurements = ", nmeas)
    end

    return E_lap_mean, E_lap_err
end


"""
Metropolis VMC that equilibrates both positions and spin labels.
Currently in progress, missing t in acceptance probability.
"""
function run_metropolis_flips(k11::Float64, delta11::Float64,
                       k22::Float64, delta22::Float64,
                       k12::Float64, delta12::Float64,
                       t::Float64,
                       h::Float64=0.0;
                       N1::Int, N2::Int, seed::Int=1234, verbose::Bool=true,
                       use_spin_flips::Bool=true,
                       debug_flips::Bool=false,
                       debug_stride::Int=100,
                       equilibration_steps::Int=VMC_EQUILIBRATION_STEPS,
                       production_steps::Int=VMC_PRODUCTION_STEPS)
    equilibration_steps >= 0 || throw(ArgumentError("equilibration_steps must be nonnegative"))
    production_steps > 0 || throw(ArgumentError("production_steps must be positive"))
    total_steps = equilibration_steps + production_steps
    N_total = N1 + N2
    n = 1.0
    L = N_total / n
    invL = 1.0 / L

    rng = MersenneTwister(seed)

    # Initialize positions and spins (Float64)
    x = rand(rng, N_total) .* L
    spin = zeros(Float64, N_total)
    for i in 1:N1
        spin[i] = 1.0
    end
    for i in N1+1:N_total
        spin[i] = 2.0
    end

    # Pre-allocate buffers
    F = zeros(N_total)
    logR_acc = zeros(N_total)

    logpsi = log_psiT(x, spin, k11, delta11, k22, delta22, k12, delta12; N_total, L, invL)

    E_lap_acc = 0.0
    E_lap_acc2 = 0.0
    P2_acc = 0.0
    nmeas = 0
    dbg_flips_attempted = 0
    dbg_flips_accepted  = 0

    do_flips = use_spin_flips && t != 0.0

    Nbins_corr = 100
    dr = (L/2) / Nbins_corr
    k_min = 2π / L
    Nk_vals = 100
    Cs      = zeros(Float64, Nbins_corr)
    Cs_count = zeros(Int, Nbins_corr)
    Cn      = zeros(Float64, Nbins_corr)
    Sk_acc  = zeros(Float64, Nk_vals)
    Sn_acc  = zeros(Float64, Nk_vals)
    Sk_count = 0

    for step in 1:total_steps
        if step % 10000 == 0
            println("Step ", step, "/", total_steps)
        end

        # Attempt position move for a random particle
        i = rand(rng, 1:N_total)
        x_new = copy(x)
        x_new[i] = pbc(x_new[i] + step_size * (rand(rng) - 0.5), L, invL)

        logpsi_new = log_psiT(x_new, spin, k11, delta11, k22, delta22, k12, delta12; N_total, L, invL)
        acc = min(1.0, exp(2.0 * (logpsi_new - logpsi)))

        if rand(rng) < acc
            x = x_new
            logpsi = logpsi_new
        end

        # Metropolis spin flip moves every step
        kinetic = 0.0; tunnel = 0.0; field = 0.0
        energy_fresh = false

        if do_flips
            kinetic, tunnel, field = turbo_compute_drift_and_energy!(F, logR_acc, x, spin,
                        k11, delta11, k22, delta22, k12, delta12, t, h;
                        N_total, L, invL)
            energy_fresh = true

            # Single-particle Metropolis flip: pick one random particle and attempt to flip it.
            i_flip = rand(rng, 1:N_total)
            p_flip = min(1.0, exp(2.0 * logR_acc[i_flip]))
            dbg_flips_attempted += 1
            if rand(rng) < p_flip
                spin[i_flip] = ifelse(spin[i_flip] == 1.0, 2.0, 1.0)
                dbg_flips_accepted += 1
                logpsi = log_psiT(x, spin, k11, delta11, k22, delta22, k12, delta12; N_total, L, invL)
                energy_fresh = false
            end

            if debug_flips && step % debug_stride == 0
                n1_dbg = sum(s -> s == 1.0 ? 1 : 0, spin)
                P_dbg  = (2 * n1_dbg - N_total) / N_total
                rate   = dbg_flips_attempted > 0 ? dbg_flips_accepted / dbg_flips_attempted : 0.0
                @printf("[step %6d] N1=%2d N2=%2d  P=%.4f  logR=[%.3f..%.3f]  flip_rate=%.3f\n",
                        step, n1_dbg, N_total-n1_dbg, P_dbg,
                        minimum(logR_acc), maximum(logR_acc), rate)
                dbg_flips_attempted = 0
                dbg_flips_accepted  = 0
            end
        end

        # Measurements after equilibration
        if step > equilibration_steps
            if !energy_fresh
                kinetic, tunnel, field = turbo_compute_drift_and_energy!(F, logR_acc, x, spin,
                            k11, delta11, k22, delta22, k12, delta12, t, h;
                            N_total, L, invL)
            end
            E_lap = kinetic + tunnel + field
            E_lap_acc += E_lap
            E_lap_acc2 += E_lap^2
            if do_flips
                n1_count = 0
                @inbounds for i in 1:N_total
                    n1_count += ifelse(spin[i] == 1.0, 1, 0)
                end
                P = (2 * n1_count - N_total) / N_total
                P2_acc += P^2
            end
            nmeas += 1

            # Cs(r) and Cn(r)
            for i in 1:N_total-1
                si = (spin[i] == 1.0) ? 1.0 : -1.0
                xi = x[i]
                for j in i+1:N_total
                    sj = (spin[j] == 1.0) ? 1.0 : -1.0
                    xj = x[j]
                    dx = xi - xj
                    dx -= L * round(dx * invL)
                    r = abs(dx)
                    if r < L/2
                        bin = Int(floor(r / dr)) + 1
                        if bin <= Nbins_corr
                            Cs[bin]       += si * sj
                            Cs_count[bin] += 1
                            Cn[bin]       += 1.0
                        end
                    end
                end
            end

            # Sk(k) and Sn(k)
            for k_idx in 1:Nk_vals
                k_val = (k_idx - 1) * k_min
                re_spin = 0.0; im_spin = 0.0
                re_dens = 0.0; im_dens = 0.0
                for i in 1:N_total
                    s  = (spin[i] == 1.0) ? 1.0 : -1.0
                    xi = x[i]
                    re_spin += s * cos(k_val * xi)
                    im_spin += s * sin(k_val * xi)
                    re_dens += cos(k_val * xi)
                    im_dens += sin(k_val * xi)
                end
                Sk_acc[k_idx] += (re_spin^2 + im_spin^2) / N_total
                Sn_acc[k_idx] += (re_dens^2 + im_dens^2) / N_total
            end
            Sk_count += 1
        end
    end

    E_lap_mean = E_lap_acc / nmeas
    E_lap_var = E_lap_acc2 / nmeas - E_lap_mean^2
    E_lap_var = max(0.0, E_lap_var)
    E_lap_err = sqrt(E_lap_var / nmeas)
    P2_mean = do_flips ? P2_acc / nmeas : 0.0

    mkpath("VMC_flips")
    open("VMC_flips/results.txt", "w") do f
        @printf(f, "N1                  %d\n", N1)
        @printf(f, "N2                  %d\n", N2)
        @printf(f, "E_VMC               %.15f\n", E_lap_mean)
        @printf(f, "E_VMC_err           %.6e\n", E_lap_err)
        @printf(f, "E_VMC_per_particle  %.15f\n", E_lap_mean / N_total)
        @printf(f, "P2                  %.15f\n", P2_mean)
        @printf(f, "equilibration_steps %d\n", equilibration_steps)
        @printf(f, "production_steps    %d\n", production_steps)
    end

    # Normalize and save correlators / structure factors
    Cs_full = zeros(Float64, Nbins_corr)
    for i in 1:Nbins_corr
        if Cs_count[i] > 0
            Cs_full[i] = Cs[i] / Cs_count[i]
        end
    end
    expected_total_pairs = nmeas * (N_total * (N_total - 1) / 2.0) * (dr / (L/2))
    Cn_full = Cn ./ expected_total_pairs
    Sk_avg  = Sk_count > 0 ? Sk_acc ./ Sk_count : zeros(Float64, Nk_vals)
    Sn_avg  = Sk_count > 0 ? Sn_acc ./ Sk_count : zeros(Float64, Nk_vals)

    base = @sprintf("N1%d_N2%d_t%.4f_h%.4f_vmc_flips", N1, N2, t, h)
    N_folder = "N$N_total"
    mkpath("VMC_flips/correlators/$N_folder")
    mkpath("VMC_flips/spin_structure_factor/$N_folder")
    mkpath("VMC_flips/density_structure_factor/$N_folder")
    open("VMC_flips/correlators/$N_folder/Cs_full_$base.txt", "w") do f
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cs_full[i])
        end
    end
    open("VMC_flips/correlators/$N_folder/Cn_full_$base.txt", "w") do f
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cn_full[i])
        end
    end
    open("VMC_flips/spin_structure_factor/$N_folder/Sk_$base.txt", "w") do f
        println(f, "# k  S_s(k)")
        for k_idx in 1:Nk_vals
            k_val = (k_idx - 1) * k_min
            println(f, k_val, " ", Sk_avg[k_idx])
        end
    end
    open("VMC_flips/density_structure_factor/$N_folder/Sn_$base.txt", "w") do f
        println(f, "# k  S_n(k)")
        for k_idx in 1:Nk_vals
            k_val = (k_idx - 1) * k_min
            println(f, k_val, " ", Sn_avg[k_idx])
        end
    end

    if verbose
        println("VMC result (unified spin-based):")
        println("Energy (via Laplacian) = ", E_lap_mean/L, " ± ", E_lap_err/L)
        if do_flips
            println("<P²>                  = ", P2_mean)
        end
        println("Measurements = ", nmeas)
    end

    return E_lap_mean, E_lap_err, P2_mean
end

