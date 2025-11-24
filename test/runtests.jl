using Test
using OptimKit
using LinearAlgebra
using Random: Random
Random.seed!(1234)

# Test linesearches
@testset "Linesearch" for (fg, x₀) in [(x -> (sin(x) + x^4, cos(x) + 4 * x^3), 0.0),
                                       (x -> (x^2, 2 * x), 1.0),
                                       (x -> (exp(x) - x^3, exp(x) - 3 * x^2), 3.9), # should trigger bisection
                                       (x -> (exp(x) - x^3, exp(x) - 3 * x^2), 2), # should trigger infinities
                                       (x -> (sum(x .^ 2), 2 * x), [1.0, 2.0]),
                                       (x -> (2 * x[1]^2 + x[2]^4 - x[3]^2 + x[3]^4,
                                              [4 * x[1], 4 * x[2]^3, -2 * x[3] + 4 * x[3]^3]),
                                        [1.0, 2.0, 3.0])]
    f₀, g₀ = fg(x₀)
    for i in 1:100
        c₁ = 0.5 * rand()
        c₂ = 0.5 + 0.5 * rand()

        ls = HagerZhangLineSearch(; c₁=c₁, c₂=c₂, ϵ=0, ρ=1.5, maxfg=100, maxiter=100)
        x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; verbosity=4)
        @test f ≈ fg(x)[1]
        @test g ≈ fg(x)[2]
        @test ξ == -g₀
        @test dot(ξ, g) >= c₂ * dot(ξ, g₀)
        @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2 * c₁ - 1) * dot(ξ, g₀) > dot(ξ, g)

        x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; initialguess=1e-4, verbosity=2) # test extrapolation phase
        @test f ≈ fg(x)[1]
        @test g ≈ fg(x)[2]
        @test ξ == -g₀
        @test dot(ξ, g) >= c₂ * dot(ξ, g₀)
        @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2 * c₁ - 1) * dot(ξ, g₀) > dot(ξ, g)

        x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; initialguess=1e4) # test infinities
        @test f ≈ fg(x)[1]
        @test g ≈ fg(x)[2]
        @test ξ == -g₀
        @test dot(ξ, g) >= c₂ * dot(ξ, g₀)
        @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2 * c₁ - 1) * dot(ξ, g₀) > dot(ξ, g)
    end
end

function quadraticproblem(B, y)
    function fg(x)
        g = B * (x - y)
        f = dot(x - y, g) / 2
        return f, g
    end
    return fg
end

function quadratictupleproblem(B, y)
    function fg(x)
        x1, x2 = x
        y1, y2 = y
        g1 = B * (x1 - y1)
        g2 = x2 - y2
        f = dot(x1 - y1, g1) / 2 + (x2 - y2)^2 / 2
        return f, (g1, g2)
    end
    return fg
end

algorithms = (GradientDescent, ConjugateGradient, LBFGS)

@testset "Optimization Algorithm $algtype" for algtype in algorithms
    n = 10
    y = randn(n)
    A = randn(n, n)
    A = A' * A
    fg = quadraticproblem(A, y)
    x₀ = randn(n)
    alg = algtype(; verbosity=2, gradtol=1e-12, maxiter=10_000_000)
    x, f, g, numfg, normgradhistory = optimize(fg, x₀, alg)
    @test x ≈ y rtol = cond(A) * 1e-12
    @test f < 1e-12

    n = 1000
    y = randn(n)
    U, S, V = svd(randn(n, n))
    smax = maximum(S)
    A = U * Diagonal(1 .+ S ./ smax) * U'
    # well conditioned, all eigenvalues between 1 and 2
    fg = quadratictupleproblem(A' * A, (y, 1.0))
    x₀ = (randn(n), 2.0)
    alg = algtype(; verbosity=3, gradtol=1e-8)
    x, f, g, numfg, normgradhistory = optimize(fg, x₀, alg)
    @test x[1] ≈ y rtol = 1e-7
    @test x[2] ≈ 1 rtol = 1e-7
    @test f < 1e-12
end

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(OptimKit)
end
