using Printf

# ====== SLURM settings (shared across all runs) ======
const CPUS_PER_TASK = 112
const TIME_LIMIT    = "24:00:00"
const ACCOUNT       = "lab_upc59"
const QOS           = "res_upc59_a"
const PARTITION     = "high-cpu"
const PREFIX        = "tg_check"

function generate_folders(;
    g::Float64,
    t::Float64,
    dt::Float64,
    N1::Int,
    N2::Int,
    num_steps::Int,
    num_walkers::Int,
    forward_walk_steps::Int,
    time_order::Int,
    h_values::Vector{Float64},
    g12_values::Vector{Float64},
    parent_folder::String="",
    estimate::Bool=true,
    method::String="dmc_pure",
    measurement_stride::Int=1_000,
    equilibration_steps::Int=50_000)

    method in ("dmc_mixed", "dmc_pure", "vmc", "dmc_pure_ovlp") ||
        throw(ArgumentError("method must be dmc_mixed, dmc_pure, vmc, or dmc_pure_ovlp"))
    measurement_stride > 0 || throw(ArgumentError("measurement_stride must be positive"))
    equilibration_steps >= 0 || throw(ArgumentError("equilibration_steps must be nonnegative"))
    method != "vmc" && equilibration_steps >= num_steps &&
        throw(ArgumentError("equilibration_steps must be less than num_steps for DMC methods"))

    N_total = N1 + N2
    count = 0

    for h in h_values, g12 in g12_values
        folder_name = @sprintf("%s_g%.6f_g12%.6f_h%.6f_t%.6f_fwd%d_N%d", PREFIX, g, g12, h, t, forward_walk_steps, N_total)
        folder = isempty(parent_folder) ? folder_name : joinpath(parent_folder, folder_name)
        mkpath(folder)
        runner_path = relpath(joinpath(@__DIR__, "run_dmc_cc_forward.jl"), abspath(folder))

        if estimate
            # MF estimate of N1, N2 based on polarization P^2 = 1 - (2t/(g12-g))^2
            denom = g12 - g
            P_mf = denom > 2*t ? sqrt(1.0 - (2*t/denom)^2) : 0.0
            N1_use = round(Int, N_total * (1 + P_mf) / 2)
            N2_use = N_total - N1_use
        else
            N1_use = N1
            N2_use = N2
        end

        # Write params.dat
        open(joinpath(folder, "params.dat"), "w") do f
            println(f, "N1                  $N1_use")
            println(f, "N2                  $N2_use")
            println(f, "method              $method")
            println(f, "measurement_stride  $measurement_stride")
            println(f, "equilibration_steps $equilibration_steps")
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
            @printf(f,  "#SBATCH --job-name=crit_exp_N%d_g12%.6f_h%.6f_t%.6f_fwd%d\n", N_total, g12, h, t, forward_walk_steps)
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
            println(f, "julia --project=~ -t \$SLURM_CPUS_PER_TASK $runner_path 2>&1 | tee run.log")
        end

        # Write run_local.sh
        open(joinpath(folder, "run_local.sh"), "w") do f
            println(f, "#!/bin/bash")
            println(f, "julia -t auto $runner_path 2>&1 | tee run.log")
        end

        println("Created: $folder")
        count += 1
    end

    return count
end

if abspath(PROGRAM_FILE) == @__FILE__

    h_values           = [0.000]
    dt                 = 0.01
    num_steps          = 8_200_000
    num_walkers        = 2000
    forward_walk_steps = 100_000
    time_order         = 2
    method             = "dmc_pure"
    measurement_stride = 1_000
    equilibration_steps = 50_000
    estimate           = true
    parent_folder      = ""
    sizes              = [(40,40)]
    g_values           = [10.]
    t_fracs            = [0.15]  # as multiples of g

    local total = 0
    for g in g_values, t_frac in t_fracs
        t       = t_frac*g
        t = 0.023
        # g12c    = g + 2*t
        # g12min = 0.85*g12c
        # g12max = 1.3*g12c
        
        # Npoints    = 25
        # g12_values = collect(range(g12min, g12max, length=Npoints))
        g12_values = [11.0]

        for (N1, N2) in sizes
            total += generate_folders(;
                g, t, dt, N1, N2,
                num_steps, num_walkers, forward_walk_steps, time_order,
                h_values, g12_values, parent_folder, estimate, method, measurement_stride,
                equilibration_steps)
        end
    end

    println("Total folders created: $total")
end

