abstract type CGFlavor
end

function cg(fg, x; linesearch = HagerZhangLineSearch(),
                    flavor::CGFlavor = HagerZhang(),
                    retract = _retract, inner = _inner, transport = _transport,
                    scale! = _scale!, add! = _add!,
                    maxiter = typemax(Int), gradtol = 1e-8, verbosity::Int = 0)

    f, g = fg(x)
    normgrad = sqrt(inner(x, g, g))
    normgradhistory = [normgrad]
    d = scale!(deepcopy(g), -1)
    α = 1e-2

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("CG: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while numiter < maxiter
        numiter += 1
        xprev = x
        dprev = d
        gprev = g

        x, f, g, α, = linesearch(fg, x, d, (f, g);
            initialguess = α, retract = retract, inner = inner, verbosity = verbosity-1)
        normgrad = sqrt(inner(x, g, g))
        push!(normgradhistory, normgrad)
        if normgrad <= gradtol
            break
        end
        # next search direction
        gprev = transport(gprev, xprev, dprev, α)
        dprev = transport(dprev, xprev, dprev, α)
        y = add!(deepcopy(g), gprev, -1)
        β = let x = x
            flavor(g, gprev, y, dprev, (d1,d2)->inner(x,d1,d2))
        end
        d = add!(scale!(deepcopy(g), -1), dprev, β)

        verbosity >= 2 &&
            @info @sprintf("CG: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, β = %.2e",
                            numiter, f, normgrad, α, β)
    end
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("CG: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e",
                            numiter, f, normgrad)
        else
            @warn @sprintf("CG: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                            f, normgrad)
        end
    end
    return x, f, g, normgradhistory
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
