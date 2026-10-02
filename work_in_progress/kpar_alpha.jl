
using Printf

function kpar_from_a(a::Float64)
    gamma = abs(2.0 / a)
    pi = 3.141592653589793

    if gamma > 8.0
        Kpar = 1.0 + 4.0 / gamma
    else
        Kpar = pi / sqrt(gamma) / sqrt(1.0 - sqrt(gamma) / (2.0 * pi))
    end

    return Kpar
end

function alpha_from_a(a::Float64)
    Kpar = kpar_from_a(a)
    return 1.0 / Kpar
end

function print_kpar_alpha(a::Float64)
    Kpar = kpar_from_a(a)
    alpha = alpha_from_a(a)
    @printf("a = %.12f\n", a)
    @printf("gamma = %.12f\n", abs(2.0 / a))
    @printf("Kpar = %.12f\n", Kpar)
    @printf("alpha = 1/Kpar = %.12f\n", alpha)
end

