# Linesearch from Algorithm 851: CG_DESCENT
# (Hager & Zhang, ACM Transactions on Mathematical Software, Vol 32 (2006))
abstract type AbstractLineSearch end

struct LineSearchPoint{T<:Real,X,G}
    α::T
    ϕ::T
    dϕ::T
    x::X
    f::T # equal to ϕ
    ∇f::G
end

function checkapproxwolfe(x::LineSearchPoint, x₀::LineSearchPoint, δ, σ, ϵ)
    return (x.ϕ <= x.ϕ + ϵ) && ((2*δ-1)*x₀.dϕ >= x.dϕ >= σ*x₀.dϕ)
end
function checkexactwolfe(x::LineSearchPoint, x₀::LineSearchPoint, δ, σ)
    return (x.ϕ <= x₀.ϕ + δ*x.α*x₀.dϕ )&& (x.dϕ > σ*x₀.dϕ)
end

struct HagerZhangLineSearch{T<:Real} <: AbstractLineSearch
    δ::T # parameter for approximate first wolfe condition
    σ::T # parameter for second wolf condition
    ϵ::T # parameter for approximate Wolfe termination
    θ::T # used in update rules for bracketing interval
    γ::T # determines when a bisection step is performed
    ρ::T # used in determining initial bracketing interval
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
    return LineSearchPoint(α, ϕ, dϕ, x, f, ∇f)
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
    !(a.α < αc < b.α) && return a, b # U0
    fmax = iter.x₀.f + iter.parameters.ϵ
    c = takestep(iter, αc)
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
            @error "bisection failure: b.α - a.α = $(b.α - a.α), a.ϕ = $(a.ϕ), b.ϕ = $(b.ϕ)"
        end

        αc = (1 - θ) * a.α + θ * b.α
        c = takestep(iter, αc)
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
    while true
        c = takestep(iter, α)
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
    return (a.x, a.f, a.∇f, a.α, a.dϕ), (a, b, false)
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
            return (B.x, B.f, B.∇f, B.α, B.dϕ), (a, b, true)
        end
        αc = secant(b.α, B.α, b.dϕ, B.dϕ)
        a, b = update(iter, A, B, αc)
    elseif αc == A.α
        if checkexactwolfe(A, x₀, δ, σ) || checkapproxwolfe(A, x₀, δ, σ, ϵ)
            return (A.x, A.f, A.∇f, A.α, A.dϕ), (a, b, true)
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
        return (a.x, a.f, a.∇f, a.α, a.dϕ), (a, b, true)
    elseif bwolfe
        return (b.x, b.f, b.∇f, b.α, b.dϕ), (a, b, true)
    else
        return (a.x, a.f, a.∇f, a.α, a.dϕ), (a, b, false)
    end
end

HagerZhangLineSearch(; δ::Real = .1, σ::Real = .9, ϵ::Real = 1e-6,
                        θ::Real = 1/2, γ::Real = 2/3, ρ::Real = 5.) =
    HagerZhangLineSearch(promote(δ, σ, ϵ, θ, γ, ρ)...)

function (ls::HagerZhangLineSearch)(fg, x₀, d₀, (f0, g0) = fg(x₀);
                    retract = _retract, inner = _inner,
                    initialguess = 1., maxiter = typemax(Int),
                    verbosity::Int = 0)

    p = LineSearchPoint(zero(f0), f0, inner(x₀, g0, d₀), x₀, f0, g0)
    iter = HagerZhangLineSearchIterator(fg, retract, inner, p, d₀, initialguess, ls)
    next = iterate(iter)
    @assert next !== nothing
    k = 1
    while true
        (x, f, g, α, dϕ), state = next
        a, b, done = state
        verbosity >= 2 &&
            @info @sprintf("Linesearch %2d: [a,b] = [%.2e, %.2e], a.dϕ = %.2e, b.dϕ = %.2e, a.ϕ - ϕ₀ = %.2e, b.ϕ - ϕ₀ = %.2e", k, a.α, b.α, a.dϕ, b.dϕ, a.ϕ - f0, b.ϕ - f0)
        next = iterate(iter, state)
        k += 1
        if next === nothing || k == maxiter
            return x, f, g, α, dϕ
        end
    end
end
