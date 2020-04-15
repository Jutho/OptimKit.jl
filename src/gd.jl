struct GradientDescent{T<:Real,L<:AbstractLineSearch} <: OptimizationAlgorithm
    maxiter::Int
    gradtol::T
    linesearch::L
    verbosity::Int
end
GradientDescent(; maxiter = typemax(Int), gradtol::Real = 1e-8,
        verbosity::Int = 0,
        linesearch::AbstractLineSearch = HagerZhangLineSearch(;verbosity = verbosity - 2)) =
    GradientDescent(maxiter, gradtol, linesearch, verbosity)

function optimize(fg, x, alg::GradientDescent; precondition = _precondition,
                    retract = _retract, inner = _inner, transport! = _transport!,
                    scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    normgrad = sqrt(inner(x, g, g))
    normgradhistory = [normgrad]
    η = scale!(precondition(x, g), -1)
    α = 1e-2
    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("GD: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while true
        # new search direction
        η = scale!(precondition(x, deepcopy(g)), -1)

        # perform line search
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
            initialguess = 1.1α, retract = retract, inner = inner)
        numfg += nfg
        normgrad = sqrt(inner(x, g, g))
        push!(normgradhistory, normgrad)
        numiter += 1

        # check stopping criteria and print info
        if normgrad <= alg.gradtol || numiter >= alg.maxiter
            break
        end
        verbosity >= 2 &&
            @info @sprintf("GD: iter %4d: f = %.12f, ‖∇f‖ = %.4e, step size = %.2e",
                            numiter, f, normgrad, α)
    end
    if verbosity > 0
        if normgrad <= alg.gradtol
            @info @sprintf("GD: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e",
                            numiter, f, normgrad)
        else
            @warn @sprintf("GD: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                            f, normgrad)
        end
    end
    return x, f, g, numfg, normgradhistory
end
