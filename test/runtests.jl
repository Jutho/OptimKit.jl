using Test
using Revise
using OptimKit
using LinearAlgebra

# Test linesearches
@testset "Linesearch" begin
    for (fg, x₀) in [(x->(sin(x) + x^4, cos(x) + 4*x^3), 0.),
                        (x->(x^2, 2*x), 1.),
                        (x->(exp(x)-x^3, exp(x)-3*x^2), 2.),
                        (x->(sum(x.^2), 2*x), [1.,2.]),
                        (x->(2*x[1]^2 + x[2]^4 - x[3]^2 + x[3]^4, [4*x[1], 4*x[2]^3, -2*x[3]+4*x[3]^3]), [1.,2.,3.])]
        f₀, g₀ = fg(x₀)
        for i = 1:100
            global c₁ = 0.5*rand()
            global c₂ = 0.5 + 0.5*rand()

            ls = HagerZhangLineSearch(; c₁ = c₁, c₂ = c₂, ϵ = 0, ρ = 1.5)
            x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀)

            @test f ≈ fg(x)[1]
            @test g ≈ fg(x)[2]
            @test ξ == -g₀
            @test dot(ξ,g) >= c₂*dot(ξ,g₀)
            @test f <= f₀ + α * c₁ * dot(ξ, g₀)

            x, f, g, ξ, α, numfg = ls(fg, x₀, -g₀; initialguess = 1e-4) # test extrapolation phase

            @test f ≈ fg(x)[1]
            @test g ≈ fg(x)[2]
            @test ξ == -g₀
            @test dot(ξ,g) >= c₂*dot(ξ,g₀)
            @test f <= f₀ + α * c₁ * dot(ξ, g₀) || (2*c₁ - 1)*dot(ξ,g₀) > dot(ξ,g)
        end
    end
end






# const n = 100
#
# algorithms = (GradientDescent, ConjugateGradient, LBFGS)
#
# function problem1(B, y)
#     function fg(x)
#         g = B*(x-y)
#         f = dot(x-y, g)/2
#         return f, g
#     end
#     return fg
# end
#
# function rescaleproblem(fg, λ, μ)
#     return function (x)
#         f, g = fg(x/μ)
#         λ*f, λ*g/μ
#     end
# end
#
#
# @testset for algtype in algorithms
#     y = randn(n)
#     A = randn(n, n)
#     fg = problem1(A'*A, y)
#     x₀ = randn(n)
#     alg = algtype()
#     x, f, g, numfg, normgradhistory = optimize(fg, x₀, alg)
#
#     λ = 4*rand()
#     μ = 1
#     fg2 = rescaleproblem(fg, λ, μ)
#     alg2 = algtype(; gradtol = alg.gradtol*λ/μ)
#     x2, f2, g2, numfg2, normgradhistory2 = optimize(fg2, μ*x₀, alg2)
#
#     @test abs(numfg - numfg2) <= 5
#     numiter = length(normgradhistory)
#     numiter2 = length(normgradhistory2)
#     @test abs(numiter - numiter2) <= 2
#     k = min(length(normgradhistory), length(normgradhistory2), n)
#     @test (λ/μ)*normgradhistory[1:k] ≈ normgradhistory2[1:k]
# end
