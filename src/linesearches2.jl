using LineSearches: LineSearches

struct LineSearchWrapper{A <: LineSearches.AbstractLineSearch} <: OptimKit.AbstractLineSearch
    alg::A
end

function make_ϕdϕ(fg, retract, inner, x₀, η₀)
    function ϕ(α)
        x, ξ = retract(x₀, η₀, α)
        f, ∇f = fg(x)
        return f
    end
    function dϕ(α)
        x, ξ = retract(x₀, η₀, α)
        f, ∇f = fg(x)
        return inner(x, ∇f, ξ)
    end
    function ϕdϕ(α)
        x, ξ = retract(x₀, η₀, α)
        f, ∇f = fg(x)
        return f, inner(x, ∇f, ξ)
    end
    return ϕ, dϕ, ϕdϕ
end

function (ls::LineSearchWrapper)(
        fg, x₀, η₀, fg₀ = fg(x₀);
        retract = OptimKit._retract, inner = OptimKit._inner,
        initialguess::Real = one(fg₀[1]),
        acceptfirst::Bool = false
    )
    ϕ, dϕ, ϕdϕ = make_ϕdϕ(fg, retract, inner, x₀, η₀)
    ϕ₀ = fg₀[1]
    dϕ₀ = inner(x₀, fg₀[2], η₀)
    α, _ = ls.alg(ϕ, dϕ, ϕdϕ, initialguess, ϕ₀, dϕ₀)
    x, ξ = retract(x₀, η₀, α)
    f, g = fg(x)
    numfg = 0
    return x, f, g, ξ, α, numfg
end
