"""
    struct SimpleIteration{T <: Real} <: FixedPointAlgorithm
    SimpleIteration(;
                    maxiter::Int=MAXITER[], # 1_000_000
                    gradtol::Real=GRADTOL[], # 1e-8
                    verbosity::Int=VERBOSITY[]) # 1

Simple iteration for fixed point problems.

## Parameters
- `maxiter::Int`: The maximum number of iterations.
- `gradtol::Real`: The tolerance for the norm of the residual.
- `verbosity::Int`: The verbosity level of the optimization algorithm.

The verbosity level use the following scheme:
- 0: no output
- 1: only warnings upon non-convergence
- 2: convergence information at the end of the algorithm
- 3: progress information after each iteration
"""
struct SimpleIteration{T <: Real} <: FixedPointAlgorithm
    maxiter::Int
    gradtol::T
    verbosity::Int
end
function SimpleIteration(;
               maxiter::Int=MAXITER[],
               gradtol::Real=GRADTOL[],
               verbosity::Int=VERBOSITY[])
    return SimpleIteration(maxiter, gradtol, verbosity)
end


function fixedpoint(
        fp, x₀, alg::SimpleIteration;
        (finalize!) = _finalize!,
        shouldstop = DefaultShouldStop(alg.maxiter),
        hasconverged = DefaultHasConverged(alg.gradtol),
        retract = _retract, invretract = _invretract,
        inner = _inner, (transport!) = _transport!,
        (scale!) = _scale!, (add!) = _add!,
        isometrictransport = (transport! == _transport! && inner == _inner)
    )

    t₀ = time()
    verbosity = alg.verbosity
    x = x₀
    fx = fp(x)
    numfp = 1
    numiter = 0
    g = invretract(x, fx) # residual, i.e. direction from x to fx, similar to F(x) - x in the vector case
    x, _, g = finalize!(x, false, g, numiter)
    innergg = inner(x, g, g)
    normg = sqrt(innergg)
    normghistory = [normg]
    verbosity >= 2 &&
        @info @sprintf("SimpleIteration: initializing with ‖x - F(x)‖ = %.4e", normg)
    t = time() - t₀
    _hasconverged = hasconverged(x, false, g, normg)
    _shouldstop = shouldstop(x, false, g, numfp, numiter, t)

    while !(_hasconverged || _shouldstop)
        told = t
        x = fx
        fx = fp(x)
        numfp += 1
        numiter += 1
        g = invretract(x, fx) # residual, i.e. direction from x to fx, similar to F(x) - x in the vector case
        x, _, g = finalize!(x, false, g, numiter)
        innergg = inner(x, g, g)
        normg = sqrt(innergg)
        push!(normghistory, normg)
        t = time() - t₀
        Δt = t - told
        _hasconverged = hasconverged(x, false, g, normg)
        _shouldstop = shouldstop(x, false, g, numfp, numiter, t)
        # check stopping criteria and print info
        if _hasconverged || _shouldstop
            break
        end
        verbosity >= 3 &&
            @info @sprintf("SimpleIteration: iter %4d, Δt %s: ‖f(x) - x‖ = %.4e", numiter, format_time(Δt), normg)
    end
    if _hasconverged
        verbosity >= 2 &&
            @info @sprintf("SimpleIteration: converged after %d iterations and time %s: ‖f(x) - x‖ = %.4e", numiter, format_time(t), normg)
    else
        verbosity >= 1 &&
            @warn @sprintf("SimpleIteration: not converged to requested tol after %d iterations and time %s: ‖f(x) - x‖ = %.4e", numiter, format_time(t), normg)
    end
    return x, g, numfp, normghistory
end
