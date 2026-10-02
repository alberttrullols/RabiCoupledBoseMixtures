include("dmc_cc_pure_pc.jl")
include("vmc.jl")
if !isdefined(@__MODULE__, :DMC_pure_pc_overlapping)
    include("dmc_cc_pure_pc_overlapping.jl")
end
if !isdefined(@__MODULE__, :alpha_from_a)
    include("kpar_alpha.jl")
end


function run_dmc_2comp_forward(N1::Int, N2::Int, g11::Float64, g22::Float64, g12::Float64;
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
    forward_walk_steps::Int=200,
    method::String="dmc_pure",
    measurement_stride::Int=1_000,
    acum_mixed::Bool=true,
    equilibration_steps::Int=0,
    vmc_equilibration_steps::Int=20_000,
    vmc_production_steps::Int=80_000,
    seed::Int=1234,
    alpha11::Union{Float64, Nothing}=nothing,
    alpha22::Union{Float64, Nothing}=nothing,
    alpha12::Union{Float64, Nothing}=nothing)

    method in ("dmc_pure", "dmc_pure_ovlp") ||
        throw(ArgumentError("method must be dmc_pure or dmc_pure_ovlp"))
    forward_walk_steps > 0 || throw(ArgumentError("forward_walk_steps must be positive"))
    measurement_stride > 0 || throw(ArgumentError("measurement_stride must be positive"))
    0 <= equilibration_steps < num_steps ||
        throw(ArgumentError("equilibration_steps must be between 0 and num_steps - 1"))

    N_total = N1 + N2
    n = 1.
    L = N_total/n 

    a11 = -(2.0 * hbar^2) / (m * g11)
    a22 = -(2.0 * hbar^2) / (m * g22)
    a12 = -(2.0 * hbar^2) / (m * g12)

    # Luttinger exponents for the piecewise trial wavefunction (override via alpha11/22/12 kwargs)
    alpha11 = something(alpha11, alpha_from_a(a11))
    alpha22 = something(alpha22, alpha_from_a(a22))
    alpha12 = something(alpha12, alpha_from_a(a12))

    if verbose
        @printf("Effective Luttinger parameters: K11 = %.6f (alpha11 = %.6f), K22 = %.6f (alpha22 = %.6f), K12 = %.6f (alpha12 = %.6f)\n",
            1.0 / alpha11, alpha11, 1.0 / alpha22, alpha22, 1.0 / alpha12, alpha12)
    end

    # Compute E_T from VMC if not provided
    if E_T_init === nothing && compute_E_T
        if verbose
            println("Computing trial energy E_T from VMC...")
            println("Running VMC with parameters: ")
            println("t = ", t)
            println("h = ", h)
        end
        E_T_init, _, _ = run_metropolis_flips(t, h;
            N1=N1, N2=N2, vmc_equilibration_steps=vmc_equilibration_steps, vmc_production_steps=vmc_production_steps, verbose=verbose,
            alpha11, alpha22, alpha12, a11, a22, a12)
    elseif E_T_init === nothing
        error("E_T_init not provided and compute_E_T is false. Please provide E_T_init or set compute_E_T=true")
    end

    if verbose
        println("\n" * "="^50)
        println("Running DMC with forward walking ($method) with parameters:")
        println("  N1, N2 = ", N1, ", ", N2)
        println("  g11, g22, g12 = ", g11, ", ", g22, ", ", g12)
        println("  num_steps = ", num_steps)
        println("  num_walkers = ", num_walkers)
        println("  dt = ", dt)
        println("  E_T = ", E_T_init)
        println("  forward_walk_steps = ", forward_walk_steps)
        println("  threads = ", Threads.nthreads())
        println("="^50 * "\n")
    end

    dmc_kwargs = (
        importance_sampling=importance_sampling,
        branching=branching,
        time_order=time_order,
        verbose=verbose,
        reblocking=reblocking,
        use_full_flip_expression=use_full_flip_expression,
        forward_walk_steps=forward_walk_steps,
        acum_mixed=acum_mixed,
        equilibration_steps=equilibration_steps,
        seed=seed,
    )

    if method == "dmc_pure_ovlp"
        return DMC_pure_pc_overlapping(
            num_steps, num_walkers, dt, E_T_init,
            alpha11, alpha22, alpha12, a11, a22, a12,
            t, N1, N2, h;
            dmc_kwargs..., measurement_stride=measurement_stride)
    end

    return DMC_pure_pc(
        num_steps, num_walkers, dt, E_T_init,
        alpha11, alpha22, alpha12, a11, a22, a12,
        t, N1, N2, h;
        dmc_kwargs...)
end


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

if abspath(PROGRAM_FILE) == @__FILE__
    p = read_params("params.dat")

    method             = get(p, "method", "dmc_pure")
    method in ("dmc_mixed", "dmc_pure", "vmc", "dmc_pure_ovlp") ||
        error("Unsupported method '$method'. Choose dmc_mixed, dmc_pure, vmc, or dmc_pure_ovlp.")

    N1                 = parse(Int,     p["N1"])
    N2                 = parse(Int,     p["N2"])
    g11                = parse(Float64, p["g11"])
    g22                = parse(Float64, p["g22"])
    g12                = parse(Float64, p["g12"])
    t                  = parse(Float64, p["t"])
    h                  = parse(Float64, p["h"])
    dt                 = parse(Float64, p["dt"])
    num_steps          = parse(Int,     p["num_steps"])
    num_walkers        = parse(Int,     p["num_walkers"])
    time_order         = parse(Int,     p["time_order"])
    measurement_stride = parse(Int, get(p, "measurement_stride", "1000"))
    acum_mixed         = parse(Bool, get(p, "acum_mixed", "true"))
    equilibration_steps = parse(Int, get(p, "equilibration_steps", "0"))
    vmc_equilibration_steps = parse(Int, get(p, "vmc_equilibration_steps", "20000"))
    vmc_production_steps = parse(Int, get(p, "vmc_production_steps", "80000"))
    alpha11            = haskey(p, "alpha11") ? parse(Float64, p["alpha11"]) : nothing
    alpha22            = haskey(p, "alpha22") ? parse(Float64, p["alpha22"]) : nothing
    alpha12            = haskey(p, "alpha12") ? parse(Float64, p["alpha12"]) : nothing

    if method in ("dmc_pure", "dmc_pure_ovlp")
        forward_walk_steps = parse(Int, p["forward_walk_steps"])
        run_dmc_2comp_forward(N1, N2, g11, g22, g12;
            t, h, dt,
            num_steps, num_walkers,
            forward_walk_steps,
            method,
            measurement_stride,
            acum_mixed,
            equilibration_steps,
            time_order,
            alpha11, alpha22, alpha12,
            vmc_equilibration_steps,
            vmc_production_steps,
        )
    elseif method == "dmc_mixed"
        include("run_dmc_mixed.jl")
        run_dmc_2comp_mixed(N1, N2, g11, g22, g12;
            t, h, dt,
            num_steps, num_walkers,
            equilibration_steps,
            vmc_equilibration_steps,
            vmc_production_steps,
            time_order,
            alpha11, alpha22, alpha12,
        )
    else
        include("run_vmc.jl")
        run_vmc_from_params(p)
    end
end
