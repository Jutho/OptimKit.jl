struct LBFGS{T<:Real,L<:AbstractLineSearch} <: OptimizationAlgorithm
    m::Int
    maxiter::Int
    gradtol::T
    linesearch::L
    verbosity::Int
end
LBFGS(m::Int = 8; maxiter = typemax(Int), gradtol::Real = 1e-8,
        verbosity::Int = 0,
        linesearch::AbstractLineSearch = HagerZhangLineSearch(;verbosity = verbosity - 2)) =
    LBFGS(m, maxiter, gradtol, linesearch, verbosity)

function optimize(fg, x, alg::LBFGS; retract = _retract, inner = _inner,
                    transport! = _transport!, scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    normgrad = sqrt(inner(x, g, g))
    normgradhistory = [normgrad]
    η = scale!(deepcopy(g), -1)

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("LBFGS: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)

    # Iteration 1: set up H
    numiter += 1
    xprev = x
    ηprev = η
    gprev = g

    x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
        initialguess = 1., retract = retract, inner = inner)
    numfg += nfg
    normgrad = sqrt(inner(x, g, g))
    push!(normgradhistory, normgrad)

    # set up InverseHessian approximation
    if isometrictransport
        # use trick from A BROYDEN CLASS OF QUASI-NEWTON METHODS FOR RIEMANNIAN OPTIMIZATION: define new isometric transport such that, applying it to transported ηprev, it returns a vector proportional to ξ
        gprev = transport!(gprev, xprev, ηprev, α, x)
        normη = sqrt(inner(xprev, ηprev, ηprev))
        normξ = sqrt(inner(x, ξ, ξ))
        β = normη/normξ
        ξ₁ = transport!(deepcopy(ηprev), xprev, ηprev, α, x)
        ξ₂ = scale!(ξ, normη/normξ)
        ν₁ = add!(ξ₁, ξ₂, +1)
        ν₂ = scale!(deepcopy(ξ₂), -2)
        gprev = add!(gprev, ν₁, -2*inner(x, ν₁, gprev)/inner(x, ν₁, ν₁))
        gprev = add!(gprev, ν₂, -2*inner(x, ν₂, gprev)/inner(x, ν₂, ν₂))
        ηprev = ξ₂
        # ξ₁ = add!(ξ₁, ν₁, -2*inner(x, ν₁, ξ₁)/inner(x, ν₁, ν₁))
        # ξ₁ = add!(ξ₁, ν₂, -2*inner(x, ν₂, ξ₁)/inner(x, ν₂, ν₂))
        # @show ξ₁, ξ₂
        y = add!(scale!(deepcopy(g), 1/β), gprev, -1)
        s = scale!(ηprev, α)
        ρ = 1/inner(x, s, y)
    else # scaled transport, make sure previous direction does not grow in norm
        gprev = transport!(gprev, xprev, ηprev, α, x)
        normη = sqrt(inner(xprev, ηprev, ηprev))
        normξ = sqrt(inner(x, ξ, ξ))
        ηprev = ξ
        if normξ > normη
            β = normη/normξ
            ηprev = scale!(ηprev, β)
            y = add!(scale!(deepcopy(g), 1/β), gprev, -1)
        else
            y = add!(deepcopy(g), gprev, -1)
        end
        s = scale!(ηprev, α)
        ρ = 1/inner(x, s, y)
    end
    @assert ρ > 0
    m = alg.m
    H = LBFGSInverseHessian(m, [s], [y], [ρ])
    η = let x = x
        scale!(H(g, (ξ1, ξ2)->inner(x, ξ1, ξ2), add!, scale!), -1)
    end

    verbosity >= 2 &&
        @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, m = %d",
                        numiter, f, normgrad, α, length(H))

    while numiter < alg.maxiter && normgrad > alg.gradtol
        numiter += 1
        xprev = x
        gprev = g
        ηprev = η

        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
            initialguess = 1., acceptfirst = true, retract = retract, inner = inner)
        numfg += nfg
        normgrad = sqrt(inner(x, g, g))
        push!(normgradhistory, normgrad)
        if normgrad <= alg.gradtol
            break
        end
        # next search direction
        if isometrictransport
            # use trick from A BROYDEN CLASS OF QUASI-NEWTON METHODS FOR RIEMANNIAN OPTIMIZATION: define new isometric transport such that, applying it to transported ηprev, it returns a vector proportional to ξ
            gprev = transport!(gprev, xprev, ηprev, α, x)
            for k = length(H):-1:1
                @inbounds s, y, ρ = H[k]
                s = transport!(s, xprev, ηprev, α, x)
                y = transport!(y, xprev, ηprev, α, x)
                H[k] = (s, y, ρ)
            end
            normη = sqrt(inner(xprev, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            β = normη/normξ
            ξ₁ = transport!(ηprev, xprev, ηprev, α, x)
            ξ₂ = ξ
            if ξ₁ !== ξ
                ξ₂ = scale!(ξ, normη/normξ)
                ν₁ = add!(deepcopy(ξ₁), ξ₂, +1)
                ν₂ = scale!(deepcopy(ξ₂), -2)
                gprev = add!(gprev, ν₁, -2*inner(x, ν₁, gprev)/inner(x, ν₁, ν₁))
                gprev = add!(gprev, ν₂, -2*inner(x, ν₂, gprev)/inner(x, ν₂, ν₂))
                for k = length(H):-1:1
                    @inbounds s, y, ρ = H[k]
                    s = add!(s, ν₁, -2*inner(x, ν₁, s)/inner(x, ν₁, ν₁))
                    s = add!(s, ν₂, -2*inner(x, ν₂, s)/inner(x, ν₂, ν₂))
                    y = add!(y, ν₁, -2*inner(x, ν₁, y)/inner(x, ν₁, ν₁))
                    y = add!(y, ν₂, -2*inner(x, ν₂, y)/inner(x, ν₂, ν₂))
                    H[k] = (s, y, ρ)
                end
            end
            ηprev = ξ₂
            y = add!(scale!(deepcopy(g), 1/β), gprev, -1)
            s = scale!(ηprev, α)
            ρ = 1/inner(x, s, y)
            @assert ρ > 0 # should be the case because of wolfe
            push!(H, (s, y, ρ))
        else # scaled transport, make sure previous direction does not grow in norm
            gprev = transport!(gprev, xprev, ηprev, α, x)
            for k = length(H):-1:1
                @inbounds s, y, ρ = H[k]
                s = transport!(s, xprev, ηprev, α, x)
                y = transport!(y, xprev, ηprev, α, x)
                ρ = 1/inner(x, s, y)
                if ρ < 0
                    for j = 1:k
                        popfirst!(H)
                    end
                    break
                end
                @inbounds H[k] = (s, y, ρ)
            end
            normη = sqrt(inner(xprev, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))
            ηprev = ξ
            if normξ > normη
                β = normη/normξ
                ηprev = scale!(ηprev, β)
                y = add!(scale!(deepcopy(g), 1/β), gprev, -1)
            else
                y = add!(deepcopy(g), gprev, -1)
            end
            s = scale!(ηprev, α)
            ρ = 1/inner(x, s, y)
            if ρ < 0
                empty!(H)
            else
                push!(H, (s, y, ρ))
            end
        end
        η = let x = x
            scale!(H(g, (ξ1, ξ2)->inner(x, ξ1, ξ2), add!, scale!), -1)
        end

        verbosity >= 2 &&
            @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, m = %d",
                            numiter, f, normgrad, α, length(H))
    end
    if verbosity > 0
        if normgrad <= alg.gradtol
            @info @sprintf("LBFGS: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e",
                            numiter, f, normgrad)
        else
            @warn @sprintf("LBGFS: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                            f, normgrad)
        end
    end
    return x, f, g, numfg, normgradhistory
end

mutable struct LBFGSInverseHessian{T1,T2,T3}
    maxlength::Int
    length::Int
    first::Int
    S::Vector{T1}
    Y::Vector{T2}
    ρ::Vector{T3}
    function LBFGSInverseHessian{T1,T2,T3}(maxlength::Int, S::Vector{T1}, Y::Vector{T2}, ρ::Vector{T3}) where {T1, T2, T3}
        @assert length(S) == length(Y) == length(ρ)
        l = length(S)
        S = resize!(copy(S), maxlength)
        Y = resize!(copy(Y), maxlength)
        ρ = resize!(copy(ρ), maxlength)
        return new{T1,T2,T3}(maxlength, l, 1, S, Y, ρ)
    end
end
LBFGSInverseHessian(maxlength::Int, S::Vector{T1}, Y::Vector{T2}, ρ::Vector{T3}) where {T1, T2, T3} = LBFGSInverseHessian{T1, T2, T3}(maxlength, S, Y, ρ)

Base.length(H::LBFGSInverseHessian) = H.length

@inline function Base.getindex(H::LBFGSInverseHessian, i::Int)
    @boundscheck if i < 1 || i > H.length
        throw(BoundsError(H, i))
    end
    n = H.maxlength
    idx = H.first + i - 1
    idx = ifelse(idx > n, idx - n, idx)
    return (getindex(H.S, idx), getindex(H.Y, idx), getindex(H.ρ, idx))
end

@inline function Base.setindex!(H::LBFGSInverseHessian, (s, y, ρ), i)
    @boundscheck if i < 1 || i > H.length
        throw(BoundsError(H, i))
    end
    n = H.maxlength
    idx = H.first + i - 1
    idx = ifelse(idx > n, idx - n, idx)
    (setindex!(H.S, s, idx), setindex!(H.Y, y, idx), setindex!(H.ρ, ρ, idx))
end

@inline function Base.push!(H::LBFGSInverseHessian, (s, y, ρ))
    if H.length < H.maxlength
        H.length += 1
    else
        H.first = (H.first == H.maxlength ? 1 : H.first + 1)
    end
    @inbounds setindex!(H, (s, y, ρ), H.length)
    return H
end
@inline function Base.pop!(H::LBFGSInverseHessian)
    @inbounds v = H[H.length]
    H.length -= 1
    return v
end
@inline function Base.popfirst!(H::LBFGSInverseHessian)
    @inbounds v = H[1]
    H.first = (H.first == H.maxlength ? 1 : H.first + 1)
    H.length -= 1
    return v
end

@inline function Base.empty!(H::LBFGSInverseHessian)
    H.length = 0
    H.first = 1
    return H
end

function (H::LBFGSInverseHessian)(g, inner, add!, scale!; α = similar(H.ρ))
    length(H) == 0 && return deepcopy(g)
    q = deepcopy(g)
    for k = length(H):-1:1
        s, y, ρ = H[k]
        α[k] = ρ * inner(s, q)
        q = add!(q, y, -α[k])
    end
    s, y, ρ = H[length(H)]
    γ = inner(s,y)/inner(y,y)
    z = scale!(q, γ)
    for k = 1:length(H)
        s, y, ρ = H[k]
        β = ρ * inner(y, z)
        z = add!(z, s, (α[k]-β))
    end
    return z
end
