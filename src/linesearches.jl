# Linesearch from Algorithm 851: CG_DESCENT
# (Hager & Zhang, ACM Transactions on Mathematical Software, Vol 32 (2006))
abstract type AbstractLineSearch end

struct LineSearchPoint{T<:Real,X,G}
    α::T # step length
    ϕ::T # local function value
    dϕ::T # local directional derivative of cost function: dϕ/dα
    x::X
    f::T # equal to ϕ
    ∇f::G # gradient in point
    ξ::G # local tangent at linesearch path: dϕ = inner(x, ξ, ∇f)
end

function checkapproxwolfe(x::LineSearchPoint, x₀::LineSearchPoint, c₁, c₂, ϵ)
    return (x.ϕ <= x₀.ϕ + ϵ) && ((2*c₁-1)*x₀.dϕ >= x.dϕ >= c₂*x₀.dϕ)
end
function checkexactwolfe(x::LineSearchPoint, x₀::LineSearchPoint, c₁, c₂)
    return (x.ϕ <= x₀.ϕ + c₁*x.α*x₀.dϕ )&& (x.dϕ > c₂*x₀.dϕ)
end

struct HagerZhangLineSearch{T<:Real} <: AbstractLineSearch
    c₁::T # parameter for (approximate) first wolfe condition: c₁ < 1/2 < c₂
    c₂::T # parameter for second wolf condition: c₁ < 1/2 < c₂
    ϵ::T # parameter for approximate Wolfe termination
    θ::T # used in update rules for bracketing interval
    γ::T # determines when a bisection step is performed
    ρ::T # used in determining initial bracketing interval
    maxiter::Int
    verbosity::Int
end


struct HagerZhangLineSearchIterator{T₁<:Real,F₁,F₂,F₃,X,G,T₂<:Real}
    fdf::F₁ # computes function value and gradient for a given x, i.e. f, g, extra = f(x, oldextra...)
    retract::F₂ # function used to step in direction η₀ with step size α, i.e. x, ξ = retract(x₀, η₀, α) where x = Rₓ₀(α*η₀) is the new position and ξ = D Rₓ₀(α*η₀)[η₀] is the derivative or tangent of x to α at the position x
    inner::F₃ # function used to compute inner product between gradient and direction, i.e. dϕ = inner(x, g, d); can depend on x (i.e. metric on a manifold)
    p₀::LineSearchPoint{T₁,X,G} # initial position, containing x₀, f₀, g₀
    η₀::G # search direction
    α₀::T₁ # initial guess for step size
    acceptfirst::Bool # whether or not the initial guess can be accepted (e.g. LBFGS)
    parameters::HagerZhangLineSearch{T₂}
end

function takestep(iter, α)
    x, ξ = iter.retract(iter.p₀.x, iter.η₀, α)
    f, ∇f = iter.fdf(x)
    ϕ = f
    dϕ = iter.inner(x, ∇f, ξ)
    return LineSearchPoint(α, ϕ, dϕ, x, f, ∇f, ξ)
end

secant(a, b, fa, fb) = (a*fb-b*fa)/(fb-fa)
# function secant2(iter::HagerZhangLineSearchIterator, a::LineSearchPoint, b::LineSearchPoint)
#     # interval has (a.dϕ < 0, a.ϕ <= f₀+ϵ), (b.dϕ >= 0)
#     αc = secant(a.α, b.α, a.dϕ, b.dϕ)
#     A, B = update(iter, a, b, αc)
#     if αc == B.α
#         αc = secant(b.α, B.α, b.dϕ, B.dϕ)
#         return update(iter, A, B, αc)
#     elseif αc == A.α
#         αc = secant(a.α, A.α, a.dϕ, A.dϕ)
#         return update(iter, A, B, αc)
#     else
#         return A, B
#     end
# end

function update(iter::HagerZhangLineSearchIterator, a::LineSearchPoint, b::LineSearchPoint, αc)
    # interval has (a.dϕ < 0, a.ϕ <= f₀+ϵ), (b.dϕ >= 0)
    p₀ = iter.p₀
    c₁ = iter.parameters.c₁
    c₂ = iter.parameters.c₂
    ϵ = iter.parameters.ϵ
    ϕ₀ = iter.p₀.f
    fmax = ϕ₀ + iter.parameters.ϵ
    !(a.α < αc < b.α) && return a, b, 0 # U0
    c = takestep(iter, αc)
    @assert isfinite(c.ϕ)
    iter.parameters.verbosity > 2 &&
        @info @sprintf("Linesearch update: try c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕ₀ = %.2e", c.α, c.dϕ, c.ϕ - ϕ₀)
    if checkexactwolfe(c, p₀, c₁, c₂) || checkapproxwolfe(c, p₀, c₁, c₂, ϵ)
        return c, c, 1
    end
    if c.dϕ >= zero(c.dϕ) # U1
        return a, c, 1
    elseif c.dϕ < zero(c.dϕ) && c.ϕ <= fmax # U2
        return c, b, 1
    else # U3
        a, b, nfg = bisect(iter, a, c)
        return a, b, nfg + 1
    end
end

function bisect(iter::HagerZhangLineSearchIterator, a::LineSearchPoint, b::LineSearchPoint)
    # applied when (a.dϕ < 0, a.ϕ <= f₀+ϵ), (b.dϕ < 0, b.ϕ > f₀+ϵ)
    θ = iter.parameters.θ
    p₀ = iter.p₀
    c₁ = iter.parameters.c₁
    c₂ = iter.parameters.c₂
    ϵ = iter.parameters.ϵ
    fmax = p₀.f + ϵ
    numfg = 0
    while true
        if b.α - a.α < eps()
            @warn @sprintf("Linesearch bisection failure: [a, b] = [%.2e, %.2e], b-a = %.2e, dϕᵃ = %.2e, dϕᵇ = %.2e, (ϕᵇ - ϕᵃ)/(b-a) = %.2e", a.α, b.α, b.α - a.α, a.dϕ, b.dϕ, (b.ϕ - a.ϕ)/(b.α - a.α))
            return a, b, numfg
        end
        αc = (1 - θ) * a.α + θ * b.α
        c = takestep(iter, αc)
        numfg += 1
        if iter.parameters.verbosity > 2
            @info @sprintf(
            """Linesearch bisect: [a, b] = [%.2e, %.2e], b-a = %.2e, dϕᵃ = %.2e, dϕᵇ = %.2e, (ϕᵇ - ϕᵃ) = %.2e
            ↪︎ c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕᵃ = %.2e, wolfe = %d, approxwolfe = %d""",
            a.α, b.α, b.α-a.α, a.dϕ, b.dϕ, (b.ϕ - a.ϕ), c.α, c.dϕ, c.ϕ - a.ϕ, checkexactwolfe(c, p₀, c₁, c₂), checkapproxwolfe(c, p₀, c₁, c₂, ϵ))
        end
        if checkexactwolfe(c, p₀, c₁, c₂) || checkapproxwolfe(c, p₀, c₁, c₂, ϵ)
            return c, c, numfg
        end
        if c.dϕ >= 0 # U3.a
            return a, c, numfg
        elseif c.ϕ <= fmax # U3.b
            a = c
        else # U3.c
            # a′ = takestep(iter, a.α)
            # @info @sprintf("""ϕᵃ = %.2e, ϕᵃ′ = %.2e""", a.ϕ, a′.ϕ)
            b = c
        end
    end
end

function bracket(iter::HagerZhangLineSearchIterator{T}, c::LineSearchPoint) where {T}
    numfg = 0
    p₀ = iter.p₀
    c₁ = iter.parameters.c₁
    c₂ = iter.parameters.c₂
    ϵ = iter.parameters.ϵ
    a = p₀
    fmax = a.f + ϵ
    iter.parameters.verbosity > 2 &&
        @info @sprintf("Linesearch start: dϕ₀ = %.2e, ϕ₀ = %.2e", a.dϕ, a.ϕ)
    α = c.α
    while true
        while !(isfinite(c.ϕ) && isfinite(c.dϕ))
            α = (a.α + α)/2
            c = takestep(iter, α)
            numfg += 1
        end
        if iter.parameters.verbosity > 2
            @info @sprintf("Linesearch bracket: try c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕ₀ = %.2e, wolfe = %d, approxwolfe = %d", c.α, c.dϕ, c.ϕ - p₀.ϕ, checkexactwolfe(c, p₀, c₁, c₂), checkapproxwolfe(c, p₀, c₁, c₂, ϵ))
        end
        c.dϕ >= 0 && return a, c, numfg# B1
        # from here: c.dϕ < 0
        if c.ϕ > fmax # B2
            a, b, nfg = bisect(iter, iter.p₀, c)
            return a, b, numfg + nfg
        else# B3
            a = c
            α *= iter.parameters.ρ
            c = takestep(iter, α)
            numfg += 1
            # if checkexactwolfe(c, p₀, c₁, c₂) || checkapproxwolfe(c, p₀, c₁, c₂, ϵ)
            #     return c, c, numfg
            # end
        end
    end
end


function Base.iterate(iter::HagerZhangLineSearchIterator)
    c₁ = iter.parameters.c₁
    c₂ = iter.parameters.c₂
    ϵ = iter.parameters.ϵ
    p₀ = iter.p₀
    a = takestep(iter, iter.α₀)
    if iter.acceptfirst
        if checkexactwolfe(a, p₀, c₁, c₂) || checkapproxwolfe(a, p₀, c₁, c₂, ϵ)
            return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, a, 1, true)
        end
    end
    a, b, numfg = bracket(iter, a)
    numfg += 1 # from takestep few lines above
    if a.α == b.α
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, true)
    elseif (b.α - a.α) < eps(one(a.α))
        @warn "Linesearch bracket converged to a point without satisfying Wolfe conditions?"
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, true)
    else
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, false)
    end
end

function Base.iterate(iter::HagerZhangLineSearchIterator, state::Tuple{LineSearchPoint,LineSearchPoint,Int,Bool})
    c₁ = iter.parameters.c₁
    c₂ = iter.parameters.c₂
    ϵ = iter.parameters.ϵ
    p₀ = iter.p₀

    a, b, numfg, done = state
    if done
        return nothing
    end
    dα = b.α - a.α
    # secant2 step
    αc = secant(a.α, b.α, a.dϕ, b.dϕ)
    A, B, nfg = update(iter, a, b, αc)
    numfg += nfg
    if A.α == B.α
        return (A.x, A.f, A.∇f, A.ξ, A.α, A.dϕ), (A, B, numfg, true)
    end
    if αc == B.α
        αc = secant(b.α, B.α, b.dϕ, B.dϕ)
        a, b, nfg = update(iter, A, B, αc)
        numfg += nfg
    elseif αc == A.α
        αc = secant(a.α, A.α, a.dϕ, A.dϕ)
        a, b, nfg = update(iter, A, B, αc)
        numfg += nfg
    else
        a, b = A, B
    end
    if a.α == b.α
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, true)
    end
    # end secant2
    if b.α - a.α > iter.parameters.γ * dα
        a, b, nfg = update(iter, a, b, (a.α + b.α)/2)
        numfg += nfg
    end
    if a.α == b.α
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, true)
    elseif (b.α - a.α) < eps(one(a.α))
        @warn "Linesearch bracket converged to a point without satisfying Wolfe conditions?"
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, true)
    else
        return (a.x, a.f, a.∇f, a.ξ, a.α, a.dϕ), (a, b, numfg, false)
    end
end

HagerZhangLineSearch(; c₁::Real = 1//10, c₂::Real = 9//10, ϵ::Real = 1//10^6,
                        θ::Real = 1//2, γ::Real = 2//3, ρ::Real = 5//1,
                        maxiter = typemax(Int), verbosity::Int = 0) =
    HagerZhangLineSearch(promote(c₁, c₂, ϵ, θ, γ, ρ)..., maxiter, verbosity)

function (ls::HagerZhangLineSearch)(fg, x₀, η₀, (f₀, g₀) = fg(x₀);
                    retract = _retract, inner = _inner,
                    initialguess = one(f₀), acceptfirst = false)

    df₀ = inner(x₀, g₀, η₀)
    if df₀ >= zero(df₀)
        error("linesearch was not given a descent direction!")
    end
    p₀ = LineSearchPoint(zero(f₀), f₀, df₀, x₀, f₀, g₀, η₀)
    iter = HagerZhangLineSearchIterator(fg, retract, inner, p₀, η₀, initialguess, acceptfirst, ls)
    next = iterate(iter)
    @assert next !== nothing
    k = 1
    while true
        (x, f, g, ξ, α, dϕ), state = next
        a, b, numfg, done = state
        if done
            ls.verbosity >= 1 &&
                @info @sprintf("Linesearch converged after %2d iterations: α = %.2e, dϕ = %.2e, ϕ - ϕ₀ = %.2e", k, α, dϕ, f - f₀)
            return x, f, g, ξ, α, numfg
        elseif k == ls.maxiter
            ls.verbosity >= 1 &&
                @info @sprintf("Linesearch not converged after %2d iterations: α = %.2e, dϕ = %.2e, ϕ - ϕ₀ = %.2e", k, α, dϕ, f - f₀)
            return x, f, g, ξ, α, numfg
        else
            ls.verbosity >= 2 &&
                @info @sprintf("Linesearch step %d: [a,b] = [%.2e, %.2e], dϕᵃ = %.2e, dϕᵇ = %.2e, ϕᵃ - ϕ₀ = %.2e, ϕᵇ - ϕ₀ = %.2e", k, a.α, b.α, a.dϕ, b.dϕ, a.ϕ - f₀, b.ϕ - f₀)
            next = iterate(iter, state)
            @assert next !== nothing
            k += 1
        end
    end
end
