abstract type CGFlavor
end

struct ConjugateGradient{F<:CGFlavor,T<:Real,L<:AbstractLineSearch} <: OptimizationAlgorithm
    flavor::F
    maxiter::Int
    gradtol::T
    linesearch::L
    restart::Int
    verbosity::Int
end
ConjugateGradient(; flavor = HagerZhang(), maxiter = typemax(Int), gradtol::Real = 1e-8,
        restart = 50, verbosity::Int = 0,
        linesearch::AbstractLineSearch = HagerZhangLineSearch(;verbosity = verbosity - 2)) =
    ConjugateGradient(flavor, maxiter, gradtol, linesearch, restart, verbosity)

function optimize(fg, x, alg::ConjugateGradient; precondition = _precondition,
                    retract = _retract, inner = _inner, transport! = _transport!,
                    scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    innergg = inner(x, g, g)
    normgrad = sqrt(innergg)
    normgradhistory = [normgrad]

    # compute here once to define initial value of α in scale-invariant way
    Pg = precondition(x, g)
    normPg = sqrt(inner(x, Pg, Pg))
    α = 1/(10*normPg) # initial guess: scale invariant

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("CG: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while true
        # compute new search direction
        if precondition === _precondition
            Pg = g
        else
            Pg = precondition(x, deepcopy(g))
        end
        η = scale!(deepcopy(Pg), -1)
        if mod(numiter, alg.restart) == 0
            β = zero(β)
        else
            β = let x = x
                alg.flavor(g, gprev, Pg, Pgprev, ηprev, (η₁,η₂)->inner(x,η₁,η₂))
            end
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
            initialguess = α, retract = retract, inner = inner)
        numfg += nfg
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(normgradhistory, normgrad)
        numiter += 1

        # check stopping criteria and print info
        if normgrad <= alg.gradtol || numiter >= alg.maxiter
            break
        end
        verbosity >= 2 &&
            @info @sprintf("CG: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, β = %.2e",
                            numiter, f, normgrad, α, β)

        # increase α for next step
        α = (11*α)/10

        # transport gprev, ηprev and vectors in Hessian approximation to x
        gprev = transport!(gprev, xprev, ηprev, α, x)
        if precondition === _precondition
            Pgprev = gprev
        else
            Pgprev = transport!(Pgprev, xprev, ηprev, α, x)
        end
        ηprev = transport!(deepcopy(ηprev), xprev, ηprev, α, x)
    end
    if verbosity > 0
        if normgrad <= alg.gradtol
            @info @sprintf("CG: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e",
                            numiter, f, normgrad)
        else
            @warn @sprintf("CG: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                            f, normgrad)
        end
    end
    return x, f, g, numfg, normgradhistory
end

struct HagerZhang{T<:Real} <: CGFlavor
    η::T
    θ::T
end
HagerZhang(; η::Real = 0.4, θ::Real = 1.0) = HagerZhang(promote(η, θ)...)

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

    β = (gPy - HZ.θ*(yPy/dy)*dg)/dy

    # small β truncation of "THE LIMITED MEMORY CONJUGATE GRADIENT METHOD" (2013)
    # requires inverse preconditioner to have inner(d, P\d) in the denominator
    # η = HZ.η*dgprev/dinvPd
    # so instead use simplified Polak-Ribiere truncation:
    return max(β, zero(β))
end

struct HestenesStiefel <: CGFlavor
    pos::Bool
end
HestenesStiefel() = HestenesStiefel(true)
function (HS::HestenesStiefel)(g, gprev, Pg, Pgprev, dprev, inner)
    # y = Pg - Pgprev : do not form exactly
    β = (inner(g, Pg) - inner(g, Pgprev))/(inner(dprev, Pg) - inner(dprev, Pgprev))
    return HS.pos ? max(zero(β), β) : β
end

struct FletcherReeves <: CGFlavor
end
function (::FletcherReeves)(g, gprev, Pg, Pgprev, dprev, inner)
    return inner(g, Pg)/inner(gprev, Pgprev)
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
    return inner(g, Pg)/(inner(dprev, Pg) - inner(dprev, Pgprev))
end
