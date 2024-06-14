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

function optimize(fg, x, alg::GradientDescent;
                    precondition = _precondition, finalize! = _finalize!,
                    retract = _retract, inner = _inner, transport! = _transport!,
                    scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    innergg = inner(x, g, g)
    normgrad = safe_sqrt(innergg)
    fhistory = [f]
    normgradhistory = [normgrad]

    # compute here once to define initial value of α in scale-invariant way
    Pg = precondition(x, g)
    normPg = safe_sqrt(inner(x, Pg, Pg))
    α = 1/(normPg) # initial guess: scale invariant

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("GD: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while true
        # compute new search direction
        Pg = precondition(x, deepcopy(g))
        η = scale!(Pg, -1) # we don't need g or Pg anymore, so we can overwrite it

        # perform line search
        _xlast[] = x # store result in global variables to debug linesearch failures
        _glast[] = g
        _dlast[] = η
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
            initialguess = α, retract = retract, inner = inner)
        numfg += nfg
        numiter += 1
        x, f, g = finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = safe_sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)

        # check stopping criteria and print info
        if normgrad <= alg.gradtol || numiter >= alg.maxiter
            break
        end
        verbosity >= 2 &&
            @info @sprintf("GD: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, nfg = %d",
                            numiter, f, normgrad, α, nfg)

        # increase α for next step
        α = 2*α
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
    history = [fhistory normgradhistory]
    return x, f, g, numfg, history
end
