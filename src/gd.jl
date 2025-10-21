"""
    GradientDescent(; 
                    maxiter::Int=MAXITER[], # 1_000_000
                    gradtol::Real=GRADTOL[], # 1e-8
                    verbosity::Int=VERBOSITY[], # 1
                    ls_maxiter::Int=LS_MAXITER[], # 10
                    ls_maxfg::Int=LS_MAXFG[], # 20
                    ls_verbosity::Int=LS_VERBOSITY[], # 1
                    linesearch = HagerZhangLineSearch(maxiter=ls_maxiter, maxfg=ls_maxfg, verbosity=ls_verbosity))


Gradient Descent optimization algorithm.

## Parameters
- `maxiter::Int`: The maximum number of iterations.
- `gradtol::T`: The tolerance for the norm of the gradient.
- `verbosity::Int`: The verbosity level of the optimization algorithm.
- `ls_maxiter::Int`: The maximum number of iterations for the line search.
- `ls_maxfg::Int`: The maximum number of function evaluations for the line search.
- `ls_verbosity::Int`: The verbosity level of the line search algorithm.
- `linesearch`: The line search algorithm to use; if a custom value is provided,
  it overrides `ls_maxiter`, `ls_maxfg`, and `ls_verbosity`.

Both `verbosity` and `ls_verbosity` use the following scheme:
- 0: no output
- 1: only warnings upon non-convergence
- 2: convergence information at the end of the algorithm
- 3: progress information after each iteration
- 4: more detailed information (only for the linesearch)
"""
struct GradientDescent{T<:Real,L<:AbstractLineSearch} <: OptimizationAlgorithm
    maxiter::Int
    gradtol::T
    verbosity::Int
    linesearch::L
end
function GradientDescent(;
                         maxiter::Int=MAXITER[],
                         gradtol::Real=GRADTOL[],
                         verbosity::Int=VERBOSITY[],
                         ls_maxiter::Int=LS_MAXITER[],
                         ls_maxfg::Int=LS_MAXFG[],
                         ls_verbosity::Int=LS_VERBOSITY[],
                         linesearch::AbstractLineSearch=HagerZhangLineSearch(;
                                                                             maxiter=ls_maxiter,
                                                                             maxfg=ls_maxfg,
                                                                             verbosity=ls_verbosity))
    return GradientDescent(maxiter, gradtol, verbosity, linesearch)
end

function optimize(fg, x, alg::GradientDescent;
                  precondition=_precondition,
                  (finalize!)=_finalize!,
                  shouldstop=DefaultShouldStop(alg.maxiter),
                  hasconverged=DefaultHasConverged(alg.gradtol),
                  retract=_retract, inner=_inner, (transport!)=_transport!,
                  (scale!)=_scale!, (add!)=_add!,
                  isometrictransport=(transport! == _transport! && inner == _inner))
    t₀ = time()
    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    numiter = 0
    innergg = inner(x, g, g)
    normgrad = sqrt(innergg)
    fhistory = [f]
    normgradhistory = [normgrad]
    t = time() - t₀
    _hasconverged = hasconverged(x, f, g, normgrad)
    _shouldstop = shouldstop(x, f, g, numfg, numiter, t)

    # compute here once to define initial value of α in scale-invariant way
    Pg = precondition(x, g)
    normPg = sqrt(inner(x, Pg, Pg))
    α = 1 / (normPg) # initial guess: scale invariant

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("GD: initializing with f = %.12e, ‖∇f‖ = %.4e", f, normgrad)
    while !(_hasconverged || _shouldstop)
        told = t
        # compute new search direction
        Pg = precondition(x, deepcopy(g))
        η = scale!(Pg, -1) # we don't need g or Pg anymore, so we can overwrite it

        # perform line search
        _xlast[] = x # store result in global variables to debug linesearch failures
        _glast[] = g
        _dlast[] = η
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
                                            initialguess=α,
                                            retract=retract, inner=inner)
        numfg += nfg
        numiter += 1
        x, f, g = finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)
        t = time() - t₀
        Δt = t - told
        _hasconverged = hasconverged(x, f, g, normgrad)
        _shouldstop = shouldstop(x, f, g, numfg, numiter, t)

        # check stopping criteria and print info
        if _hasconverged || _shouldstop
            break
        end
        verbosity >= 3 &&
            @info @sprintf("GD: iter %4d, Δt %s: f = %.12e, ‖∇f‖ = %.4e, α = %.2e, nfg = %d",
                           numiter, format_time(Δt), f, normgrad, α, nfg)

        # increase α for next step
        α = 2 * α
    end
    if _hasconverged
        verbosity >= 2 &&
            @info @sprintf("GD: converged after %d iterations and time %s: f = %.12e, ‖∇f‖ = %.4e",
                           numiter, format_time(t), f, normgrad)
    else
        verbosity >= 1 &&
            @warn @sprintf("GD: not converged to requested tol after %d iterations and time %s: f = %.12e, ‖∇f‖ = %.4e",
                           numiter, format_time(t), f, normgrad)
    end
    history = [fhistory normgradhistory]
    return x, f, g, numfg, history
end
