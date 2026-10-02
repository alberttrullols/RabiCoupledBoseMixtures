if !isdefined(@__MODULE__, :run_metropolis_flips)
    include("vmc.jl")
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

function run_vmc_from_params(params::Dict{String, String})
    N1 = parse(Int, params["N1"])
    N2 = parse(Int, params["N2"])
    g11 = parse(Float64, params["g11"])
    g22 = parse(Float64, params["g22"])
    g12 = parse(Float64, params["g12"])
    t = parse(Float64, params["t"])
    h = parse(Float64, params["h"])
    
    equilibration_steps = parse(Int, get(params, "vmc_equilibration_steps", string(VMC_EQUILIBRATION_STEPS)))
    production_steps = parse(Int, get(params, "vmc_production_steps", string(VMC_PRODUCTION_STEPS)))

    N_total = N1 + N2
    L = Float64(N_total)
    a11 = -(2.0 * hbar^2) / (m * g11)
    a22 = -(2.0 * hbar^2) / (m * g22)
    a12 = -(2.0 * hbar^2) / (m * g12)
    k11, delta11 = compute_k_delta(a11, L)
    k22, delta22 = compute_k_delta(a22, L)
    k12, delta12 = compute_k_delta(a12, L)

    println("Running VMC only with parameters from params.dat")
    return run_metropolis_flips(k11, delta11, k22, delta22, k12, delta12, t, h;
        N1, N2, vmc_equilibration_steps=equilibration_steps, vmc_production_steps=production_steps, verbose=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_vmc_from_params(read_params("params.dat"))
end