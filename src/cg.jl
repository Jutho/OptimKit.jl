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

function optimize(fg, x, alg::ConjugateGradient; retract = _retract, inner = _inner,
                    transport! = _transport!, scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    normgrad = sqrt(inner(x, g, g))
    normgradhistory = [normgrad]
    η = scale!(deepcopy(g), -1)
    α = 1e-2

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("CG: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while numiter < alg.maxiter
        numiter += 1
        xprev = x
        ηprev = η
        gprev = g

        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
            initialguess = α, retract = retract, inner = inner)
        normgrad = sqrt(inner(x, g, g))
        numfg += nfg
        push!(normgradhistory, normgrad)
        if normgrad <= alg.gradtol
            break
        end
        # next search direction
        if isometrictransport
            # use trick from A BROYDEN CLASS OF QUASI-NEWTON METHODS FOR RIEMANNIAN OPTIMIZATION: define new isometric transport such that, applying it to transported ηprev, it returns a vector proportional to ξ but with the norm of ηprev
            gprev = transport!(gprev, xprev, ηprev, α, x)
            normη = sqrt(inner(xprev, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            ξ₁ = transport!(ηprev, xprev, ηprev, α, x)
            ξ₂ = ξ
            if ξ₁ !== ξ
                # if ξ₁ == ξ, it also means normξ = normη, since the latter is preserved
                ξ₂ = scale!(ξ, normη/normξ)
                ν₁ = add!(deepcopy(ξ₁), ξ₂, +1)
                ν₂ = scale!(deepcopy(ξ₂), -2)
                gprev = add!(gprev, ν₁, -2*inner(x, ν₁, gprev)/inner(x, ν₁, ν₁))
                gprev = add!(gprev, ν₂, -2*inner(x, ν₂, gprev)/inner(x, ν₂, ν₂))
                # @show normη normξ
                # ξ₁ = add!(ξ₁, ν₁, -2*inner(x, ν₁, ξ₁)/inner(x, ν₁, ν₁))
                # ξ₁ = add!(ξ₁, ν₂, -2*inner(x, ν₂, ξ₁)/inner(x, ν₂, ν₂))
                # ξ₁ = add!(ξ₁, ξ₂, -1)
                # @show inner(x, ξ₁, ξ₁)
            end
            ηprev = ξ₂
            y = add!(scale!(deepcopy(g), normξ/normη), gprev, -1)
        else # scaled transport, make sure previous direction does not grow in norm
            gprev = transport!(gprev, xprev, ηprev, α, x)
            normη = sqrt(inner(xprev, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            ηprev = ξ
            if normξ > normη
                ηprev = scale!(ηprev, normη/normξ)
                y = add!(scale!(deepcopy(g), normξ/normη), gprev, -1)
            else
                y = add!(deepcopy(g), gprev, -1)
            end
        end
        β = let x = x
            alg.flavor(g, gprev, y, ηprev, (η₁,η₂)->inner(x,η₁,η₂))
        end
        η = scale!(deepcopy(g), -1)
        if mod(numiter, alg.restart) == 0
            β = zero(β)
        else
            η = add!(η, ηprev, β)
        end

        verbosity >= 2 &&
            @info @sprintf("CG: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, β = %.2e",
                            numiter, f, normgrad, α, β)
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

function (HZ::HagerZhang)(g, gprev, y, dprev, inner)
    dy = inner(dprev, y)
    yy = inner(y, y)
    β = (inner(y, g) - HZ.θ*yy/dy*inner(dprev, g))/dy

    η = HZ.η*inner(dprev, gprev)/inner(dprev, dprev)

    return max(β, η)
end

struct HestenesStiefel <: CGFlavor
end
(::HestenesStiefel)(g, gprev, y, dprev, inner) = inner(g, y)/inner(dprev, y)

struct FletcherReeves <: CGFlavor
end
(::FletcherReeves)(g, gprev, y, dprev, inner) = inner(g, g)/inner(gprev, gprev)

struct PolakRibierePolyak <: CGFlavor
    pos::Bool
end
PolakRibierePolyak() = PolakRibierePolyak(true)
function (PRP::PolakRibierePolyak)(g, gprev, y, dprev, inner)
    β = inner(g, y) / inner(gprev, gprev)
    return PRP.pos ? max(zero(β), β) : β
end

struct DaiYuan <: CGFlavor
end
(::DaiYuan)(g, gprev, y, dprev, inner) = inner(g, g)/inner(dprev, y)
