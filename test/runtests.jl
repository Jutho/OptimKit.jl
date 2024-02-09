using Test
using OptimKit
using LinearAlgebra

# Test linesearches
@testset "Linesearch" for (fg, x₀) in [(x->(sin(x) + x^4, cos(x) + 4*x^3), 0.),
                        (x->(x^2, 2*x), 1.),
                        (x->(exp(x)-x^3, exp(x)-3*x^2), 3.9), # should trigger bisection
                        (x->(exp(x)-x^3, exp(x)-3*x^2), 2), # should trigger infinities
                        (x->(sum(x.^2), 2*x), [1.,2.]),
                        (x->(2*x[1]^2 + x[2]^4 - x[3]^2 + x[3]^4, [4*x[1], 4*x[2]^3, -2*x[3]+4*x[3]^3]), [1.,2.,3.])]
    f₀, g₀ = fg(x₀)
    for i = 1:100
        global c₁ = 0.5*rand()
        global c₂ = 0.5 + 0.5*rand()

        ls = HagerZhangLineSearch(; c₁ = c₁, c₂ = c₂, ϵ = 0, ρ = 1.5)
        x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; verbosity = 4)

        @test f ≈ fg(x)[1]
        @test g ≈ fg(x)[2]
        @test ξ == -g₀
        @test dot(ξ,g) >= c₂*dot(ξ,g₀)
        @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2*c₁ - 1)*dot(ξ,g₀) > dot(ξ,g)

        x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; initialguess = 1e-4, verbosity = 2) # test extrapolation phase

        @test f ≈ fg(x)[1]
        @test g ≈ fg(x)[2]
        @test ξ == -g₀
        @test dot(ξ,g) >= c₂*dot(ξ,g₀)
        @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2*c₁ - 1)*dot(ξ,g₀) > dot(ξ,g)

        x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; initialguess = 1e4, verbosity = 0) # test infinities

        @test f ≈ fg(x)[1]
        @test g ≈ fg(x)[2]
        @test ξ == -g₀
        @test dot(ξ,g) >= c₂*dot(ξ,g₀)
        @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2*c₁ - 1)*dot(ξ,g₀) > dot(ξ,g)
    end
end

function quadraticproblem(B, y)
    function fg(x)
        g = B*(x-y)
        f = dot(x-y, g)/2
        return f, g
    end
    return fg
end

algorithms = (GradientDescent, ConjugateGradient, LBFGS)

@testset "Optimization Algorithm $algtype" for algtype in algorithms
    n = 10
    y = randn(n)
    A = randn(n, n)
    fg = quadraticproblem(A'*A, y)
    x₀ = randn(n)
    alg = algtype(; verbosity = 2, gradtol = 1e-12)
    x, f, g, numfg, normgradhistory = optimize(fg, x₀, alg)
    @test x ≈ y
    @test f < 1e-14

    n = 1000
    y = randn(n)
    U, S, V = svd(randn(n,n))
    smax = maximum(S)
    A = U * Diagonal(1 .+ S ./ smax ) * U'
    # well conditioned, all eigenvalues between 1 and 2
    fg = quadraticproblem(A'*A, y)
    x₀ = randn(n)
    alg = algtype(; verbosity = 2, gradtol = 1e-8)
    x, f, g, numfg, normgradhistory = optimize(fg, x₀, alg)
    @test x ≈ y rtol=1e-7
    @test f < 1e-14
end
