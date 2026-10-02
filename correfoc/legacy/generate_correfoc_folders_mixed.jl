using Printf

# ====== SLURM settings (MN5 correfoc infrastructure) ======
const CPUS_PER_TASK = 32
const TIME_LIMIT    = "24:00:00"
const ACCOUNT       = "lab_upc59"
const QOS           = "res_upc59_a"
const PARTITION     = "high-cpu"
const PREFIX        = "DMC_derivatives_uw"

function generate_folders(;
    g::Float64,
    t::Float64,
    h::Float64,
    dt::Float64,
    N1::Int,
    N2::Int,
    num_steps::Int,
    num_walkers::Int,
    forward_walk_steps::Int,
    time_order::Int,
    g12_values::Vector{Float64})

    N_total = N1 + N2
    count = 0

    for g12 in g12_values
        folder = @sprintf("%s_g%.6f_g12%.6f_h%.6f_t%.6f_fwd%d_N%d", PREFIX, g, g12, h, t, forward_walk_steps, N_total)
        mkpath(folder)

        # MF estimate of N1, N2 based on polarization P^2 = 1 - (2t/(g12-g))^2
        denom = g12 - g
        P_mf = denom > 2*t ? sqrt(1.0 - (2*t/denom)^2) : 0.0
        N1_est = round(Int, N_total * (1 + P_mf) / 2)
        N2_est = N_total - N1_est

        # Write params.dat
        open(joinpath(folder, "params.dat"), "w") do f
            println(f, "N1                  $N1_est")
            println(f, "N2                  $N2_est")
            @printf(f,  "g11                 %.6f\n", g)
            @printf(f,  "g22                 %.6f\n", g)
            @printf(f,  "g12                 %.6f\n", g12)
            @printf(f,  "t                   %.6f\n", t)
            @printf(f,  "h                   %.6f\n", h)
            @printf(f,  "dt                  %.6f\n", dt)
            println(f, "num_steps           $num_steps")
            println(f, "num_walkers         $num_walkers")
            println(f, "forward_walk_steps  $forward_walk_steps")
            println(f, "time_order          $time_order")
        end

        # Write job.sh
        open(joinpath(folder, "job.sh"), "w") do f
            println(f, "#!/bin/bash")
            @printf(f,  "#SBATCH --job-name=dmc_mixed_N%d_g12%.6f_h%.6f_t%.6f_fwd%d\n", N_total, g12, h, t, forward_walk_steps)
            println(f, "#SBATCH --chdir=.")
            println(f, "#SBATCH --output=job_%j.out")
            println(f, "#SBATCH --error=job_%j.err")
            println(f, "#SBATCH --ntasks=1")
            println(f, "#SBATCH --cpus-per-task=$CPUS_PER_TASK")
            println(f, "#SBATCH --time=$TIME_LIMIT")
            println(f, "#SBATCH --partition=$PARTITION")
            println(f, "#SBATCH --account=$ACCOUNT")
            println(f, "#SBATCH --qos=$QOS")
            println(f, "")
            println(f, "module load modulepath/EESSI/2025.06")
            println(f, "module load Julia/1.12.2")
            println(f, "julia --project=~ -t \$SLURM_CPUS_PER_TASK ../run_dmc_mixed_correfoc.jl 2>&1 | tee run.log")
        end

        # Write run_local.sh
        open(joinpath(folder, "run_local.sh"), "w") do f
            println(f, "#!/bin/bash")
            println(f, "julia -t auto ../run_dmc_mixed_correfoc.jl 2>&1 | tee run.log")
        end

        println("Created: $folder")
        count += 1
    end

    return count
end

if abspath(PROGRAM_FILE) == @__FILE__

    h                  = 0.0
    dt                 = 0.001
    num_steps          = 2_500_000
    num_walkers        = 1000
    forward_walk_steps = 30000
    time_order         = 2
    sizes              = [(25, 25)]
    g_values           = [0.01]
    t_fracs            = [0.05]  # as multiples of g

    local total = 0
    for g in g_values, t_frac in t_fracs
        t          = t_frac*g
        g12c       = g + 2*t
        g12min     = 0.8 * g12c
        g12max     = 1.2 * g12c
        Npoints    = 100
        g12_values = collect(range(g12min, g12max, length=Npoints))

        for (N1, N2) in sizes
            total += generate_folders(;
                g, t, h, dt, N1, N2,
                num_steps, num_walkers, forward_walk_steps, time_order,
                g12_values)
        end
    end

    println("Total folders created: $total")
end
