# Linesearch from Algorithm 851: CG_DESCENT
# (Hager & Zhang, ACM Transactions on Mathematical Software, Vol 32 (2006))
abstract type AbstractLineSearch end

struct LineSearchPoint{T<:Real,X,G}
    α::T # step length
    ϕ::T # local function value
    dϕ::T # local directional derivative of cost function: dϕ/dα
    x::X
    f::T # equal to ϕ
    ∇f::G
    dx::G # local tangent at linesearch path: dϕ = inner(x, dx, ∇f)
end

function checkapproxwolfe(x::LineSearchPoint, x₀::LineSearchPoint, δ, σ, ϵ)
    return (x.ϕ <= x.ϕ + ϵ) && ((2*δ-1)*x₀.dϕ >= x.dϕ >= σ*x₀.dϕ)
end
function checkexactwolfe(x::LineSearchPoint, x₀::LineSearchPoint, δ, σ)
    return (x.ϕ <= x₀.ϕ + δ*x.α*x₀.dϕ )&& (x.dϕ > σ*x₀.dϕ)
end

struct HagerZhangLineSearch{T<:Real} <: AbstractLineSearch
    δ::T # parameter for (approximate) first wolfe condition
    σ::T # parameter for second wolf condition
    ϵ::T # parameter for approximate Wolfe termination
    θ::T # used in update rules for bracketing interval
    γ::T # determines when a bisection step is performed
    ρ::T # used in determining initial bracketing interval
    maxiter::Int
    verbosity::Int
end


struct HagerZhangLineSearchIterator{T<:Real,F₁,F₂,F₃,X,G,D}
    fdf::F₁ # computes function value and gradient for a given x, i.e. f, g, extra = f(x, oldextra...)
    retract::F₂ # function used to step in direction d with step size c, i.e. x, d′ = retract(x₀, d, c) where d′ is the new direction, i.e. the derivative d x / d c of x to c at that position
    inner::F₃ # function used to compute inner product between gradient and direction, i.e. dϕ = inner(x, g, d); can depend on x (i.e. metric on a manifold)
    x₀::LineSearchPoint{T,X,G} # initial position
    d::D # search direction
    α₀::T # initial guess for step size
    parameters::HagerZhangLineSearch{T}
end

function takestep(iter, α)
    x, d = iter.retract(iter.x₀.x, iter.d, α)
    f, ∇f = iter.fdf(x)
    ϕ = f
    dϕ = iter.inner(x, ∇f, d)
    return LineSearchPoint(α, ϕ, dϕ, x, f, ∇f, d)
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
    ϕ₀ = iter.x₀.f
    fmax = ϕ₀ + iter.parameters.ϵ
    !(a.α < αc < b.α) && return a, b # U0
    c = takestep(iter, αc)
    iter.parameters.verbosity > 2 &&
        @info @sprintf("Linesearch update: try c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕ₀ = %.2e", c.α, c.dϕ, c.ϕ - ϕ₀)
    if c.dϕ > 0 # U1
        return a, c
    elseif c.dϕ < 0 && c.ϕ <= fmax # U2
        return c, b
    else # U3
        return bisect(iter, a, c)
    end
end

function bisect(iter::HagerZhangLineSearchIterator, a::LineSearchPoint, b::LineSearchPoint)
    # applied when (a.dϕ < 0, a.ϕ <= f₀+ϵ), (b.dϕ < 0, b.ϕ > f₀+ϵ)
    θ = iter.parameters.θ
    fmax = iter.x₀.f + iter.parameters.ϵ
    while true

        if b.α - a.α < eps()
            error(@sprintf("Linesearch bisection failure: [a, b] = [%.2e, %.2e], b-a = %.2e, dϕᵃ = %.2e, dϕᵇ = %.2e, (ϕᵇ - ϕᵃ)/(b-a) = %.2e", a.α, b.α, b.α - a.α, a.dϕ, b.dϕ, (b.ϕ - a.ϕ)/(b.α - a.α)))
        end

        αc = (1 - θ) * a.α + θ * b.α
        c = takestep(iter, αc)
        if iter.parameters.verbosity > 2
            @info @sprintf(
            """Linesearch bisect: [a, b] = [%.2e, %.2e], b-a = %.2e, dϕᵃ = %.2e, dϕᵇ = %.2e, (ϕᵇ - ϕᵃ) = %.2e
            ↪︎ c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕᵃ = %.2e""",
            a.α, b.α, b.α-a.α, a.dϕ, b.dϕ, (b.ϕ - a.ϕ), c.α, c.dϕ, c.ϕ - a.ϕ)
        end

        if c.dϕ >= 0 # U3.a
            return a, c
        elseif c.ϕ <= fmax # U3.b
            a = c
        else # U3.c
            b = c
        end
    end
end

function bracket(iter::HagerZhangLineSearchIterator{T}, α = one(T)) where {T}
    a = iter.x₀
    fmax = a.f + iter.parameters.ϵ
    iter.parameters.verbosity > 2 &&
        @info @sprintf("Linesearch start: dϕ₀ = %.2e, ϕ₀ = %.2e", a.dϕ, a.ϕ)
    while true
        c = takestep(iter, α)
        if iter.parameters.verbosity > 2
            @info @sprintf("Linesearch bracket: try c = %.2e, dϕᶜ = %.2e, ϕᶜ - ϕ₀ = %.2e", c.α, c.dϕ, c.ϕ - a.ϕ)
        end
        c.dϕ >= 0 && return a, c # B1
        # from here: b.dϕ < 0
        if c.ϕ > fmax # B2
            return bisect(iter, iter.x₀, c)
        else # B3
            a = c
            α *= iter.parameters.ρ
        end
    end
end


function Base.iterate(iter::HagerZhangLineSearchIterator)
    x₀ = iter.x₀
    a, b = bracket(iter, iter.α₀)
    return (a.x, a.f, a.∇f, a.dx, a.α, a.dϕ), (a, b, false)
end

function Base.iterate(iter::HagerZhangLineSearchIterator, state::Tuple{LineSearchPoint,LineSearchPoint,Bool})
    δ = iter.parameters.δ
    σ = iter.parameters.σ
    ϵ = iter.parameters.ϵ
    x₀ = iter.x₀

    a, b, done = state
    if done
        return nothing
    end
    dα = b.α - a.α
    # secant2 step
    αc = secant(a.α, b.α, a.dϕ, b.dϕ)
    A, B = update(iter, a, b, αc)
    if αc == B.α
        if checkexactwolfe(B, x₀, δ, σ) || checkapproxwolfe(B, x₀, δ, σ, ϵ)
            return (B.x, B.f, B.∇f, B.dx, B.α, B.dϕ), (a, b, true)
        end
        αc = secant(b.α, B.α, b.dϕ, B.dϕ)
        a, b = update(iter, A, B, αc)
    elseif αc == A.α
        if checkexactwolfe(A, x₀, δ, σ) || checkapproxwolfe(A, x₀, δ, σ, ϵ)
            return (A.x, A.f, A.∇f, A.dx, A.α, A.dϕ), (a, b, true)
        end
        αc = secant(a.α, A.α, a.dϕ, A.dϕ)
        a, b = update(iter, A, B, αc)
    else
        a, b = A, B
    end
    # end secant2
    if b.α - a.α > iter.parameters.γ * dα
        a, b = update(iter, a, b, (a.α + b.α)/2)
    end
    awolfe = checkexactwolfe(a, x₀, δ, σ) || checkapproxwolfe(a, x₀, δ, σ, ϵ)
    bwolfe = checkexactwolfe(b, x₀, δ, σ) || checkapproxwolfe(b, x₀, δ, σ, ϵ)

    if a.ϕ < b.ϕ && awolfe
        return (a.x, a.f, a.∇f, a.dx, a.α, a.dϕ), (a, b, true)
    elseif bwolfe
        return (b.x, b.f, b.∇f, b.dx, b.α, b.dϕ), (a, b, true)
    else
        return (a.x, a.f, a.∇f, a.dx, a.α, a.dϕ), (a, b, false)
    end
end

HagerZhangLineSearch(; δ::Real = .1, σ::Real = .9, ϵ::Real = 1e-6,
                        θ::Real = 1/2, γ::Real = 2/3, ρ::Real = 5.,
                        maxiter = typemax(Int), verbosity::Int = 0) =
    HagerZhangLineSearch(promote(δ, σ, ϵ, θ, γ, ρ)..., maxiter, verbosity)

function (ls::HagerZhangLineSearch)(fg, x₀, d₀, (f0, g0) = fg(x₀);
                    retract = _retract, inner = _inner, initialguess = 1.)

    p = LineSearchPoint(zero(f0), f0, inner(x₀, g0, d₀), x₀, f0, g0, d₀)
    iter = HagerZhangLineSearchIterator(fg, retract, inner, p, d₀, initialguess, ls)
    next = iterate(iter)
    @assert next !== nothing
    k = 1
    while true
        (x, f, g, dx, α, dϕ), state = next
        a, b, done = state
        if done
            ls.verbosity >= 1 &&
                @info @sprintf("Linesearch converged after %2d iterations: α = %.2e, dϕ = %.2e, ϕ - ϕ₀ = %.2e", k, α, dϕ, f - f0)
            return x, f, g, dx, α
        elseif k == ls.maxiter
            ls.verbosity >= 1 &&
                @info @sprintf("Linesearch not converged after %2d iterations: α = %.2e, dϕ = %.2e, ϕ - ϕ₀ = %.2e", k, α, dϕ, f - f0)
            return x, f, g, dx, α
        else
            ls.verbosity >= 2 &&
                @info @sprintf("Linesearch step %d: [a,b] = [%.2e, %.2e], dϕᵃ = %.2e, dϕᵇ = %.2e, ϕᵃ - ϕ₀ = %.2e, ϕᵇ - ϕ₀ = %.2e", k, a.α, b.α, a.dϕ, b.dϕ, a.ϕ - f0, b.ϕ - f0)
            next = iterate(iter, state)
            @assert next !== nothing
            k += 1
        end
    end
end
