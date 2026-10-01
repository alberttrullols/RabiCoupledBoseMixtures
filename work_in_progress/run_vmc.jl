if !isdefined(@__MODULE__, :run_metropolis)
    include("vmc.jl")
end

if !isdefined(@__MODULE__, :alpha_from_a)
    include("kpar_alpha.jl")
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
    equilibration_steps = parse(Int, get(params, "equilibration_steps", string(VMC_EQUILIBRATION_STEPS)))

    N_total = N1 + N2
    L = Float64(N_total)
    a11 = -(2.0 * hbar^2) / (m * g11)
    a22 = -(2.0 * hbar^2) / (m * g22)
    a12 = -(2.0 * hbar^2) / (m * g12)

    alpha11 = haskey(params, "alpha11") ? parse(Float64, params["alpha11"]) : alpha_from_a(a11)
    alpha22 = haskey(params, "alpha22") ? parse(Float64, params["alpha22"]) : alpha_from_a(a22)
    alpha12 = haskey(params, "alpha12") ? parse(Float64, params["alpha12"]) : alpha_from_a(a12)

    println("Running piecewise trial-wavefunction VMC with alpha-based parameters")
    return run_metropolis(t, h;
        N1, N2, equilibration_steps, verbose=true,
        alpha11, alpha22, alpha12,
        a11, a22, a12)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_vmc_from_params(read_params("params.dat"))
end