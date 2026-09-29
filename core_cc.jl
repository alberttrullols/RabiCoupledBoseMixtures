using Random
using Roots: find_zero, Brent
using LoopVectorization

const hbar = 1.0
const m = 1.0

# periodic boundary conditions
@inline @fastmath pbc(dx, L, invL) = dx - L * round(dx * invL)

# log of trial wavefunction (Float64 spin, branchless param lookup)
@fastmath function log_psiT(x::AbstractVector{Float64}, spin::AbstractVector{Float64},
                            k11::Float64, delta11::Float64,
                            k22::Float64, delta22::Float64,
                            k12::Float64, delta12::Float64;
                            N_total::Int, L::Float64, invL::Float64)
    s = 0.0
    @inbounds for i in 1:N_total-1
        xi = x[i]
        si = spin[i]
        for j in i+1:N_total
            dx = abs(pbc(xi - x[j], L, invL))
            sj = spin[j]
            same = ifelse(si == sj, 1.0, 0.0)
            is1 = ifelse(si == 1.0, 1.0, 0.0)
            kj = ifelse(same == 1.0, ifelse(is1 == 1.0, k11, k22), k12)
            dj = ifelse(same == 1.0, ifelse(is1 == 1.0, delta11, delta22), delta12)
            s += log(cos(kj * dx + dj))
        end
    end
    return s
end

# Combined drift force, kinetic energy, and flip ratios in single SIMD-vectorized pass.
# Inner j-loop is @turbo'd with branchless ifelse param lookup and inlined PBC.
@fastmath function turbo_compute_drift_and_energy!(
    F::AbstractVector{Float64},
    logR_acc::AbstractVector{Float64},
    x::AbstractVector{Float64},
    spin::AbstractVector{Float64},
    k11::Float64, delta11::Float64,
    k22::Float64, delta22::Float64,
    k12::Float64, delta12::Float64,
    t::Float64,
    h::Float64;
    N_total::Int, L::Float64, invL::Float64
)::Tuple{Float64, Float64, Float64}

    fill!(F, 0.0)
    lap = 0.0

    if t != 0.0
        # Path WITH flip ratio computation
        fill!(logR_acc, 0.0)

        for i in 1:N_total
            xi = x[i]
            si = spin[i]
            si_flip = ifelse(si == 1.0, 2.0, 1.0)
            fi = 0.0
            lap_i = 0.0
            lR_i = 0.0

            @turbo for j in 1:N_total
                # Inlined PBC
                dx = xi - x[j]
                dx = dx - L * floor(dx * invL + 0.5)
                absdx = abs(dx)
                s = ifelse(dx > 0.0, 1.0, -1.0)
                mask = ifelse(i == j, 0.0, 1.0)

                # Branchless param lookup
                sj = spin[j]
                same = ifelse(si == sj, 1.0, 0.0)
                is1 = ifelse(si == 1.0, 1.0, 0.0)
                kj = ifelse(same == 1.0, ifelse(is1 == 1.0, k11, k22), k12)
                dj = ifelse(same == 1.0, ifelse(is1 == 1.0, delta11, delta22), delta12)

                theta = kj * absdx + dj
                sin_v = sin(theta)
                cos_v = cos(theta)

                # Drift force
                fi += (-kj * sin_v / cos_v * s) * mask
                # Laplacian
                lap_i += (kj * kj / (cos_v * cos_v)) * mask

                # Flip ratio: interaction params if spin[i] were flipped
                same_f = ifelse(si_flip == sj, 1.0, 0.0)
                is1_f = ifelse(si_flip == 1.0, 1.0, 0.0)
                kj_f = ifelse(same_f == 1.0, ifelse(is1_f == 1.0, k11, k22), k12)
                dj_f = ifelse(same_f == 1.0, ifelse(is1_f == 1.0, delta11, delta22), delta12)
                cos_v_f = cos(kj_f * absdx + dj_f)
                lR_i += (log(abs(cos_v_f)) - log(abs(cos_v))) * mask
            end

            F[i] = fi
            logR_acc[i] = lR_i
            lap += lap_i
        end
    else
        # Fast path: NO flip ratio computation
        for i in 1:N_total
            xi = x[i]
            si = spin[i]
            fi = 0.0
            lap_i = 0.0

            @turbo for j in 1:N_total
                dx = xi - x[j]
                dx = dx - L * floor(dx * invL + 0.5)
                absdx = abs(dx)
                s = ifelse(dx > 0.0, 1.0, -1.0)
                mask = ifelse(i == j, 0.0, 1.0)

                sj = spin[j]
                same = ifelse(si == sj, 1.0, 0.0)
                is1 = ifelse(si == 1.0, 1.0, 0.0)
                kj = ifelse(same == 1.0, ifelse(is1 == 1.0, k11, k22), k12)
                dj = ifelse(same == 1.0, ifelse(is1 == 1.0, delta11, delta22), delta12)

                theta = kj * absdx + dj
                sin_v = sin(theta)
                cos_v = cos(theta)

                fi += (-kj * sin_v / cos_v * s) * mask
                lap_i += (kj * kj / (cos_v * cos_v)) * mask
            end

            F[i] = fi
            lap += lap_i
        end
    end

    # F^2 sum for kinetic energy
    F2_sum = 0.0
    @inbounds @simd for i in 1:N_total
        F2_sum += F[i] * F[i]
    end

    # Full N×N loop double-counts pairs
    kinetic_part = (hbar^2 / (2.0 * m)) * (lap - F2_sum)

    # Tunneling from pre-computed flip ratios
    tunnel_part = 0.0
    if t != 0.0
        @inbounds for i in 1:N_total
            R = exp(logR_acc[i])
            tunnel_part -= t * exp(logR_acc[i])
        end
    end

    # Field term
    field_part = 0.0
    if h != 0.0
        n1 = 0.0
        @inbounds for i in 1:N_total
            n1 += ifelse(spin[i] == 1.0, 1.0, 0.0)
        end
        field_part = -h * (2.0 * n1 - N_total)
    end

    return kinetic_part, tunnel_part, field_part
end

function compute_k_delta(a1D, L)
    tol = 1e-20
    f(delta) = (2.0 * (-delta) / L) * tan(delta) - 1.0 / a1D
    delta = find_zero(f, (tol, pi/2 - tol), Brent())
    k = -2.0 * delta / L
    return k, delta
end
