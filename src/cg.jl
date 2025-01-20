abstract type CGFlavor end

"""
    struct ConjugateGradient{F<:CGFlavor,T<:Real,L<:AbstractLineSearch} <: OptimizationAlgorithm
    ConjugateGradient(;
                      flavor::CGFlavor=HagerZhang(),
                      restart::Int=typemax(Int);
                      maxiter::Int=typemax(Int),
                      gradtol::Real=1e-8,
                      verbosity::Int=1,
                      ls_verbosity::Int=1,
                      linesearch::AbstractLineSearch=HagerZhangLineSearch())

ConjugateGradient optimization algorithm.

## Fields
- `flavor`: The flavor of the conjugate gradient algorithm (for selecting the β parameter)
- `restart::Int`: The number of iterations after which to reset the search direction.
- `maxiter::Int`: The maximum number of iterations.
- `gradtol::T`: The tolerance for the norm of the gradient.
- `linesearch::L`: The line search algorithm to use.
- `verbosity::Int`: The verbosity level of the optimization algorithm.
- `ls_verbosity::Int`: The verbosity level of the line search algorithm.

Both verbosity levels use the following scheme:
- 0: no output
- 1: only warnings upon non-convergence
- 2: convergence information at the end of the algorithm
- 3: progress information after each iteration
- 4: more detailed information (only for the linesearch)
"""
struct ConjugateGradient{F<:CGFlavor,T<:Real,L<:AbstractLineSearch} <: OptimizationAlgorithm
    flavor::F
    restart::Int
    maxiter::Int
    gradtol::T
    linesearch::L
    verbosity::Int
    ls_verbosity::Int
end
function ConjugateGradient(;
                           flavor::CGFlavor=HagerZhang(),
                           restart::Int=typemax(Int),
                           maxiter::Int=typemax(Int),
                           gradtol::Real=1e-8,
                           verbosity::Int=1,
                           ls_verbosity::Int=1,
                           linesearch::AbstractLineSearch=HagerZhangLineSearch())
    return ConjugateGradient(flavor, restart, maxiter, gradtol, linesearch, verbosity,
                             ls_verbosity)
end

function optimize(fg, x, alg::ConjugateGradient;
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
    if precondition === _precondition
        Pg = g
    else
        Pg = precondition(x, deepcopy(g))
    end
    normPg = sqrt(abs(inner(x, g, Pg)))
    α = 1 / (normPg) # initial guess: scale invariant
    # α = one(normgrad)

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("CG: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    local xprev, gprev, Pgprev, ηprev
    while !(_hasconverged || _shouldstop)
        # compute new search direction
        if precondition === _precondition
            Pg = g
        else
            Pg = precondition(x, deepcopy(g))
        end
        η = scale!(deepcopy(Pg), -1)
        if mod(numiter, alg.restart) == 0
            β = zero(α)
        else
            β = oftype(α,
                       let x = x
                           alg.flavor(g, gprev, Pg, Pgprev, ηprev,
                                      (η₁, η₂) -> inner(x, η₁, η₂))
                       end)
            η = add!(η, ηprev, β)
        end

        # store current quantities as previous quantities
        xprev = x
        gprev = g
        Pgprev = Pg
        ηprev = η

        # perform line search
        _xlast[] = x # store result in global variables to debug linesearch failures
        _glast[] = g
        _dlast[] = η
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
                                            initialguess=α, retract=retract, inner=inner,
                                            verbosity=verbosity - 2)
        numfg += nfg
        numiter += 1
        x, f, g = finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)
        t = time() - t₀
        _hasconverged = hasconverged(x, f, g, normgrad)
        _shouldstop = shouldstop(x, f, g, numfg, numiter, t)

        # check stopping criteria and print info
        if _hasconverged || _shouldstop
            break
        end
        verbosity >= 3 &&
            @info @sprintf("CG: iter %4d, time %7.2f s: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, β = %.2e, nfg = %d",
                           numiter, t, f, normgrad, α, β, nfg)

        # transport gprev, ηprev and vectors in Hessian approximation to x
        gprev = transport!(gprev, xprev, ηprev, α, x)
        if precondition === _precondition
            Pgprev = gprev
        else
            Pgprev = transport!(Pgprev, xprev, ηprev, α, x)
        end
        ηprev = transport!(deepcopy(ηprev), xprev, ηprev, α, x)

        # increase α for next step
        α = 2 * α
    end
    if _hasconverged
        verbosity >= 2 &&
            @info @sprintf("CG: converged after %d iterations and time %.2f s: f = %.12f, ‖∇f‖ = %.4e",
                           numiter, t, f, normgrad)
    else
        verbosity >= 1 &&
            @warn @sprintf("CG: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                           f, normgrad)
    end
    history = [fhistory normgradhistory]
    return x, f, g, numfg, history
end

struct HagerZhang{T<:Real} <: CGFlavor
    η::T
    θ::T
end
HagerZhang(; η::Real=4 // 10, θ::Real=1 // 1) = HagerZhang(promote(η, θ)...)

function (HZ::HagerZhang)(g, gprev, Pg, Pgprev, dprev, inner)
    dd = inner(dprev, dprev)
    dg = inner(dprev, g)
    dgprev = inner(dprev, gprev)
    gPg = inner(g, Pg)
    gprevPgprev = inner(gprev, Pgprev)
    gPgprev = inner(g, Pgprev)
    gprevPg = inner(gprev, Pg) # should probably be the same as gprevPg

    dy = dg - dgprev
    gPy = gPg - gPgprev
    yPy = gPg + gprevPgprev - gPgprev - gprevPg

    β = (gPy - HZ.θ * (yPy / dy) * dg) / dy

    # small β truncation of "THE LIMITED MEMORY CONJUGATE GRADIENT METHOD" (2013)
    # requires inverse preconditioner to have inner(d, P\d) in the denominator
    # η = HZ.η*dgprev/dinvPd
    # so instead use simplified Polak-Ribiere truncation:
    # return max(β, zero(β))
    η = HZ.η * dgprev / dd
    if β < η
        @warn "resorting to η"
    end
    return max(β, η)
end

struct HestenesStiefel <: CGFlavor
    pos::Bool
end
HestenesStiefel() = HestenesStiefel(true)
function (HS::HestenesStiefel)(g, gprev, Pg, Pgprev, dprev, inner)
    # y = Pg - Pgprev : do not form exactly
    β = (inner(g, Pg) - inner(g, Pgprev)) / (inner(dprev, Pg) - inner(dprev, Pgprev))
    return HS.pos ? max(zero(β), β) : β
end

struct FletcherReeves <: CGFlavor
end
function (::FletcherReeves)(g, gprev, Pg, Pgprev, dprev, inner)
    return inner(g, Pg) / inner(gprev, Pgprev)
end

struct PolakRibiere <: CGFlavor
    pos::Bool
end
PolakRibiere() = PolakRibiere(true)
function (PR::PolakRibiere)(g, gprev, Pg, Pgprev, dprev, inner)
    # y = Pg - Pgprev : do not form exactly
    β = (inner(g, Pg) - inner(g, Pgprev)) / inner(gprev, Pgprev)
    return PR.pos ? max(zero(β), β) : β
end

struct DaiYuan <: CGFlavor
end
function (::DaiYuan)(g, gprev, Pg, Pgprev, dprev, inner)
    # y = Pg - Pgprev : do not form exactly
    return inner(g, Pg) / (inner(dprev, Pg) - inner(dprev, Pgprev))
end
