using Random
using Roots: find_zero, Brent
using LoopVectorization

const hbar = 1.0
const m = 1.0

# periodic boundary conditions
@inline @fastmath pbc(dx, L, invL) = dx - L * round(dx * invL)

# Short-range + long-range trial wavefunction used by GEA C implementation: 
# solve for the matching radius Rpar, then build short-distance constants Atrial and Btrial, and switch to the long-distance Luttinger tail outside that matching point.
@inline @fastmath function Rpar_equation(R::Float64, alpha::Float64, a::Float64, L::Float64)
    if R <= 0.0 || R >= L / 2.0
        return NaN
    end
    disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * R / L) * tan(π * R / L)) - 1.0)
    if disc <= 0.0
        return NaN
    end
    k = sqrt(disc)
    return -k * tan(k * R + atan(1.0 / (k * a))) - alpha * (π / L) / tan(π * R / L)
end

@inline @fastmath function solve_Rpar(alpha::Float64, a::Float64, L::Float64)
    if alpha <= 0.0 || a == 0.0
        return L / 2.0
    end

    precision = 1e-12
    Rmin = 1e-8
    Rmax = L / 2.0 - 1e-8

    R = Rmin
    disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * R / L) * tan(π * R / L)) - 1.0)
    if disc <= 0.0
        y_min = NaN
    else
        k = sqrt(disc)
        y_min = -k * tan(k * R + atan(1.0 / (k * a))) - alpha * (π / L) / tan(π * R / L)
    end

    R = Rmax
    disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * R / L) * tan(π * R / L)) - 1.0)
    if disc <= 0.0
        y_max = NaN
    else
        k = sqrt(disc)
        y_max = -k * tan(k * R + atan(1.0 / (k * a))) - alpha * (π / L) / tan(π * R / L)
    end

    if abs(y_max) < precision
        return Rmax
    end
    if isfinite(y_min) && isfinite(y_max) && y_min * y_max > 0.0
        return L / 2.0
    end

    iter = 1
    while true
        R = (Rmin + Rmax) / 2.0
        disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * R / L) * tan(π * R / L)) - 1.0)
        if disc <= 0.0
            y = NaN
        else
            k = sqrt(disc)
            y = -k * tan(k * R + atan(1.0 / (k * a))) - alpha * (π / L) / tan(π * R / L)
        end

        if isfinite(y) && isfinite(y_min) && y * y_min < 0.0
            Rmax = R
            y_max = y
        else
            Rmin = R
            y_min = y
        end

        if abs(y) < precision
            return R
        end

        if iter >= 1000
            return (Rmax * y_min - Rmin * y_max) / (y_min - y_max)
        end
        iter += 1
    end
end

# Pure two-body Bethe-Peierls resonance cos(k*r+delta) spanning the whole box (0, L/2)
@inline function edge_bethe_peierls_constants(a::Float64, L::Float64)
    f(delta) = (2.0 * (-delta) / L) * tan(delta) - 1.0 / a
    tol = 1e-12
    delta = find_zero(f, (tol, π / 2 - tol), Brent())
    k = -2.0 * delta / L
    return k, L / 2.0, 1.0, -delta / k
end

@inline @fastmath function short_range_analytic_constants(alpha::Float64, a::Float64, L::Float64)
    if alpha <= 0.0 || a == 0.0
        return 0.0, L / 2.0, 1.0, 0.0
    end
    # wf.c always builds the (repulsive) trial w.f. with a negative scattering length
    a = a > 0.0 ? -a : a

    Rpar = solve_Rpar(alpha, a, L)
    if !(Rpar < L / 2.0)
        return edge_bethe_peierls_constants(a, L)
    end

    disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * Rpar / L) * tan(π * Rpar / L)) - 1.0)
    if disc <= 0.0
        return edge_bethe_peierls_constants(a, L)
    end
    k = sqrt(disc)
    if !isfinite(k)
        return edge_bethe_peierls_constants(a, L)
    end

    Btrial = -atan(1.0 / (k * a)) / k
    Atrial = (sin(π * Rpar / L))^alpha / cos(k * (Rpar - Btrial))
    if !isfinite(Btrial) || !isfinite(Atrial)
        return edge_bethe_peierls_constants(a, L)
    end

    return k, Rpar, Atrial, Btrial
end

@inline function pair_constants(si::Float64, sj::Float64,
                                constants11::NTuple{4, Float64},
                                constants22::NTuple{4, Float64},
                                constants12::NTuple{4, Float64})
    if si == sj
        return si == 1.0 ? constants11 : constants22
    end
    return constants12
end

@inline @fastmath function pair_log_and_derivative(dx::Float64,
                                                  k::Float64, delta::Float64,
                                                  alpha::Float64,
                                                  L::Float64,
                                                  a::Float64=1.0)
    r = abs(dx)
    sgn = ifelse(dx > 0.0, 1.0, -1.0)

    if alpha == 0.0
        return 0.0, 0.0, 0.0
    end

    k_match, Rpar, Atrial, Btrial = short_range_analytic_constants(alpha, a, L)
    if isfinite(k_match) && r < Rpar
        theta = k_match * (r - Btrial)
        log_term = log(abs(Atrial * cos(theta)))
        drift = -k_match * tan(theta) * sgn
        lap = k_match * k_match * (1.0 + tan(theta) * tan(theta))
        return log_term, drift, lap
    end

    theta = π * r / L
    log_term = alpha * log(sin(theta))
    drift = alpha * (π / L) * (cos(theta) / sin(theta)) * sgn
    lap = alpha * (π / L)^2 / (sin(theta) * sin(theta))
    return log_term, drift, lap
end

@fastmath function log_psiT_piecewise(x::AbstractVector{Float64}, spin::AbstractVector{Float64};
                                     N_total::Int, L::Float64, invL::Float64,
                                     alpha11::Float64=0.0, alpha22::Float64=0.0,
                                     alpha12::Float64=0.0, a11::Float64=1.0,
                                     a22::Float64=1.0, a12::Float64=1.0)
    constants11 = short_range_analytic_constants(alpha11, a11, L)
    constants22 = short_range_analytic_constants(alpha22, a22, L)
    constants12 = short_range_analytic_constants(alpha12, a12, L)
    s = 0.0
    @inbounds for i in 1:N_total-1
        xi = x[i]
        si = spin[i]
        for j in i+1:N_total
            dx = pbc(xi - x[j], L, invL)
            sj = spin[j]
            same = ifelse(si == sj, 1.0, 0.0)
            is1 = ifelse(si == 1.0, 1.0, 0.0)
            alpha = ifelse(same == 1.0, ifelse(is1 == 1.0, alpha11, alpha22), alpha12)
            r = abs(dx)
            if alpha == 0.0
                continue
            end

            k_match, Rpar, Atrial, Btrial = pair_constants(si, sj, constants11, constants22, constants12)
            if isfinite(k_match) && r < Rpar
                s += log(abs(Atrial * cos(k_match * (r - Btrial))))
            else
                s += alpha * log(sin(π * r / L))
            end
        end
    end
    return s
end

# Combined drift force, kinetic energy, and flip ratios in single SIMD-vectorized pass.
# Uses our standard short-range Jastrow pair term and adds the asymptotic long-distance tail used in the wf.c implementation.

@fastmath function turbo_compute_drift_and_energy_piecewise!(
    F::AbstractVector{Float64},
    logR_acc::AbstractVector{Float64},
    x::AbstractVector{Float64},
    spin::AbstractVector{Float64},
    t::Float64,
    h::Float64;
    N_total::Int, L::Float64, invL::Float64,
    alpha11::Float64=0.0, alpha22::Float64=0.0,
    alpha12::Float64=0.0, a11::Float64=1.0,
    a22::Float64=1.0, a12::Float64=1.0
)::Tuple{Float64, Float64, Float64}

    fill!(F, 0.0)
    lap = 0.0
    constants11 = short_range_analytic_constants(alpha11, a11, L)
    constants22 = short_range_analytic_constants(alpha22, a22, L)
    constants12 = short_range_analytic_constants(alpha12, a12, L)

    if t != 0.0
        fill!(logR_acc, 0.0)

        for i in 1:N_total
            xi = x[i]
            si = spin[i]
            si_flip = ifelse(si == 1.0, 2.0, 1.0)
            fi = 0.0
            lap_i = 0.0
            lR_i = 0.0

            @inbounds for j in 1:N_total
                dx = xi - x[j]
                dx = dx - L * floor(dx * invL + 0.5)
                mask = ifelse(i == j, 0.0, 1.0)
                # Self-pair distance must stay away from 0 and L/2 so the masked term is finite, not Inf*0=NaN.
                absdx = ifelse(i == j, 0.25 * L, abs(dx))
                sgn = ifelse(dx > 0.0, 1.0, -1.0)

                sj = spin[j]
                same = ifelse(si == sj, 1.0, 0.0)
                is1 = ifelse(si == 1.0, 1.0, 0.0)
                alpha = ifelse(same == 1.0, ifelse(is1 == 1.0, alpha11, alpha22), alpha12)

                # Current pair contribution
                k_match, Rpar, Atrial, Btrial = pair_constants(si, sj, constants11, constants22, constants12)
                if isfinite(k_match) && absdx < Rpar
                    theta = k_match * (absdx - Btrial)
                    sin_v = sin(theta)
                    cos_v = cos(theta)
                    fi += (-k_match * sin_v / cos_v * sgn) * mask
                    lap_i += (k_match * k_match / (cos_v * cos_v)) * mask
                else
                    theta = π * absdx / L
                    fi += (alpha * (π / L) * (cos(theta) / sin(theta)) * sgn) * mask
                    lap_i += (alpha * (π / L)^2 / (sin(theta) * sin(theta))) * mask
                end

                same_f = ifelse(si_flip == sj, 1.0, 0.0)
                is1_f = ifelse(si_flip == 1.0, 1.0, 0.0)
                alpha_f = ifelse(same_f == 1.0, ifelse(is1_f == 1.0, alpha11, alpha22), alpha12)

                k_match_f, Rpar_f, Atrial_f, Btrial_f = pair_constants(si_flip, sj, constants11, constants22, constants12)
                if isfinite(k_match_f) && absdx < Rpar_f
                    theta_f = k_match_f * (absdx - Btrial_f)
                    log_ratio_f = log(abs(Atrial_f * cos(theta_f)))
                    if isfinite(k_match) && absdx < Rpar
                        lR_i += (log_ratio_f - log(abs(Atrial * cos(k_match * (absdx - Btrial))))) * mask
                    else
                        lR_i += (log_ratio_f - alpha * log(sin(π * absdx / L))) * mask
                    end
                else
                    theta_f = π * absdx / L
                    if isfinite(k_match) && absdx < Rpar
                        lR_i += (alpha_f * log(sin(theta_f)) - log(abs(Atrial * cos(k_match * (absdx - Btrial))))) * mask
                    else
                        lR_i += (alpha_f - alpha) * log(sin(theta_f)) * mask
                    end
                end
            end

            F[i] = fi
            logR_acc[i] = lR_i
            lap += lap_i
        end
    else
        for i in 1:N_total
            xi = x[i]
            si = spin[i]
            fi = 0.0
            lap_i = 0.0

            @inbounds for j in 1:N_total
                dx = xi - x[j]
                dx = dx - L * floor(dx * invL + 0.5)
                mask = ifelse(i == j, 0.0, 1.0)
                # Self-pair distance must stay away from 0 and L/2 so the masked term is finite, not Inf*0=NaN.
                absdx = ifelse(i == j, 0.25 * L, abs(dx))
                sgn = ifelse(dx > 0.0, 1.0, -1.0)

                sj = spin[j]
                same = ifelse(si == sj, 1.0, 0.0)
                is1 = ifelse(si == 1.0, 1.0, 0.0)
                alpha = ifelse(same == 1.0, ifelse(is1 == 1.0, alpha11, alpha22), alpha12)

                k_match, Rpar, Atrial, Btrial = pair_constants(si, sj, constants11, constants22, constants12)
                if isfinite(k_match) && absdx < Rpar
                    theta = k_match * (absdx - Btrial)
                    sin_v = sin(theta)
                    cos_v = cos(theta)
                    fi += (-k_match * sin_v / cos_v * sgn) * mask
                    lap_i += (k_match * k_match / (cos_v * cos_v)) * mask
                else
                    theta = π * absdx / L
                    fi += (alpha * (π / L) * (cos(theta) / sin(theta)) * sgn) * mask
                    lap_i += (alpha * (π / L)^2 / (sin(theta) * sin(theta))) * mask
                end
            end

            F[i] = fi
            lap += lap_i
        end
    end

    F2_sum = 0.0
    @inbounds @simd for i in 1:N_total
        F2_sum += F[i] * F[i]
    end

    kinetic_part = (hbar^2 / (2.0 * m)) * (lap - F2_sum)

    tunnel_part = 0.0
    if t != 0.0
        @inbounds for i in 1:N_total
            tunnel_part -= t * exp(logR_acc[i])
        end
    end

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

