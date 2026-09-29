include("dmc_cc_mixed.jl")
if !isdefined(@__MODULE__, :run_metropolis)
    include("vmc.jl")
end


function run_dmc_2comp_mixed(N1::Int, N2::Int, g11::Float64, g22::Float64, g12::Float64;
    t::Float64=0.0,
    h::Float64=0.0,
    num_steps::Int=10^5,
    num_walkers::Int=2000,
    dt::Float64=0.0001,
    E_T_init::Union{Float64, Nothing}=nothing,
    importance_sampling::Bool=true,
    branching::Bool=true,
    time_order::Int=1,
    reblocking::Bool=true,
    compute_E_T::Bool=true,
    verbose::Bool=true,
    use_full_flip_expression::Bool=true,
    equilibration_steps::Int=0,
    seed::Int=1234)

    0 <= equilibration_steps < num_steps ||
        throw(ArgumentError("equilibration_steps must be between 0 and num_steps - 1"))

    N_total = N1 + N2
    n = 1.
    L = N_total / n

    a11 = -(2.0 * hbar^2) / (m * g11)
    a22 = -(2.0 * hbar^2) / (m * g22)
    a12 = -(2.0 * hbar^2) / (m * g12)

    # k, delta imposing PBC and Bethe-Peierls BCs
    k11, delta11 = compute_k_delta(a11, L)
    k22, delta22 = compute_k_delta(a22, L)
    k12, delta12 = compute_k_delta(a12, L)

    # Compute E_T from VMC if not provided
    if E_T_init === nothing && compute_E_T
        if verbose
            println("Computing trial energy E_T from VMC...")
            println("Running VMC with parameters: ")
            println("t = ", t)
            println("h = ", h)
        end
        E_T_init, _ = run_metropolis(k11, delta11, k22, delta22, k12, delta12, t, h;
            N1=N1, N2=N2, equilibration_steps=equilibration_steps, verbose=verbose)
    elseif E_T_init === nothing
        error("E_T_init not provided and compute_E_T is false. Please provide E_T_init or set compute_E_T=true")
    end

    if verbose
        println("\n" * "="^50)
        println("Running DMC (mixed estimators) with parameters:")
        println("  N1, N2 = ", N1, ", ", N2)
        println("  g11, g22, g12 = ", g11, ", ", g22, ", ", g12)
        println("  num_steps = ", num_steps)
        println("  num_walkers = ", num_walkers)
        println("  dt = ", dt)
        println("  E_T = ", E_T_init)
        println("  threads = ", Threads.nthreads())
        println("="^50 * "\n")
    end

    # Run DMC with mixed estimators
    energy_avg, energy_err, polarization_avg, polarization_avg_err, polarization_2_avg, polarization_2_avg_err, polarization_4_avg, polarization_4_avg_err, polarization_abs_avg, polarization_abs_avg_err, energy_squared_avg = DMC_mixed(
        num_steps,
        num_walkers,
        dt,
        E_T_init,
        k11, delta11,
        k22, delta22,
        k12, delta12,
        t,
        N1, N2,
        h;
        importance_sampling=importance_sampling,
        branching=branching,
        time_order=time_order,
        verbose=verbose,
        reblocking=reblocking,
        use_full_flip_expression=use_full_flip_expression,
        equilibration_steps=equilibration_steps,
        seed=seed
    )

    return energy_avg, energy_err, polarization_avg, polarization_avg_err, polarization_2_avg, polarization_2_avg_err, polarization_4_avg, polarization_4_avg_err, polarization_abs_avg, polarization_abs_avg_err, energy_squared_avg
end


if !isdefined(@__MODULE__, :read_params)
    function read_params(filename::String="params.dat")
        params = Dict{String, String}()
        open(filename) do f
            for line in eachline(f)
                line = strip(line)
                isempty(line) && continue
                startswith(line, "#") && continue
                parts = split(line)
                length(parts) >= 2 && (params[parts[1]] = parts[2])
            end
        end
        return params
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = read_params("params.dat")

    N1          = parse(Int,     p["N1"])
    N2          = parse(Int,     p["N2"])
    g11         = parse(Float64, p["g11"])
    g22         = parse(Float64, p["g22"])
    g12         = parse(Float64, p["g12"])
    t           = parse(Float64, p["t"])
    h           = parse(Float64, p["h"])
    dt          = parse(Float64, p["dt"])
    num_steps   = parse(Int,     p["num_steps"])
    num_walkers = parse(Int,     p["num_walkers"])
    time_order  = parse(Int,     p["time_order"])
    equilibration_steps = parse(Int, get(p, "equilibration_steps", "0"))

    run_dmc_2comp_mixed(N1, N2, g11, g22, g12;
        t, h, dt,
        num_steps, num_walkers,
        equilibration_steps,
        time_order,
    )
end
