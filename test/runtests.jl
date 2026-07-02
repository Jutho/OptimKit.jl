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

@testset "LBFGS checkpoint and resume" begin
    n = 20
    y = randn(n)
    A = let B = randn(n, n); B' * B + I end
    fg = quadraticproblem(A, y)
    x₀ = randn(n)
    alg = LBFGS(; verbosity=0, gradtol=1e-12, maxiter=10_000_000)

    # Run to full convergence as ground truth
    x_full, f_full, g_full, numfg_full, history_full = optimize(fg, x₀, alg)

    # Run with early stopping after 5 iterations and collect checkpoint
    saved_states = LBFGSState[]
    checkpoint_fn = state -> push!(saved_states, state)
    stop_after_5 = (x, f, g, numfg, numiter, t) -> numiter >= 5
    converged_1e12 = (x, f, g, normgrad) -> normgrad <= 1e-12
    x_part, f_part, g_part, numfg_part, history_part =
        optimize(fg, x₀, alg; checkpoint=checkpoint_fn, shouldstop=stop_after_5,
                 hasconverged=converged_1e12)

    # Checkpoint is called once per completed iteration
    @test length(saved_states) == 5

    # Checkpoint state at iteration 5 matches optimize's returned state
    state5 = saved_states[end]
    @test state5.numiter == 5
    @test state5.x ≈ x_part
    @test state5.f ≈ f_part
    @test state5.numfg == numfg_part
    @test length(state5.fhistory) == 6      # initial + 5 iterations
    @test length(state5.normgradhistory) == 6

    # Resume from checkpoint and run to convergence; result must match full run
    x_resumed, f_resumed, g_resumed, numfg_resumed, history_resumed =
        optimize(fg, state5, alg)
    @test x_resumed ≈ x_full rtol = 1e-10
    @test f_resumed ≈ f_full rtol = 1e-10

    # Resumed history prepends the prior run's history
    @test size(history_resumed, 1) == size(history_full, 1)
    @test history_resumed[1:6, :] ≈ history_part  # first 6 rows identical to partial run

    # Resume with additional checkpoint continues counting from previous numiter
    extra_states = LBFGSState[]
    stop_after_3_more = (x, f, g, numfg, numiter, t) -> numiter >= state5.numiter + 3
    optimize(fg, state5, alg;
             checkpoint=state -> push!(extra_states, state),
             shouldstop=stop_after_3_more,
             hasconverged=converged_1e12)
    @test length(extra_states) == 3
    @test extra_states[1].numiter == 6
    @test extra_states[end].numiter == 8
end

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(OptimKit)
end
