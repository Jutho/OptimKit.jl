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
    normgrad = sqrt(inner(x, g, g))
    normgradhistory = [normgrad]
    d = scale!(deepcopy(g), -1)

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("LBFGS: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)

    # Iteration 1: set up H
    numiter += 1
    xprev = x
    dprev = d
    gprev = g

    x, f, g, dx, α = alg.linesearch(fg, x, d, (f, g);
        initialguess = 1e-2, retract = retract, inner = inner)
    normgrad = sqrt(inner(x, g, g))
    push!(normgradhistory, normgrad)

    # set up InverseHessian approximation
    gprev = transport!(gprev, xprev, dprev, α, x)
    # dprev = transport!(dprev, xprev, dprev, α, x)
    dprev = dx
    y = scale!(add!(gprev, g, -1), -1)
    s = scale!(dprev, α)
    ρ = 1/inner(x, s, y)
    @assert ρ > 0
    m = alg.m
    H = LBFGSInverseHessian(m, [s], [y], [ρ])
    d = let x = x
        scale!(H(g, (y1,y2)->inner(x, y1, y2), add!, scale!), -1)
    end

    verbosity >= 2 &&
        @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e",
                        numiter, f, normgrad, α)

    while numiter < alg.maxiter && normgrad > alg.gradtol
        numiter += 1
        xprev = x
        gprev = g
        dprev = d

        x, f, g, dx, α = alg.linesearch(fg, x, d, (f, g);
            initialguess = 1., retract = retract, inner = inner)
        normgrad = sqrt(inner(x, g, g))
        push!(normgradhistory, normgrad)
        if normgrad <= alg.gradtol
            break
        end
        # next search direction
        for k = length(H):-1:1
            @inbounds s, y, ρ = H[k]
            s = transport!(s, xprev, dprev, α, x)
            y = transport!(y, xprev, dprev, α, x)
            if !isometrictransport
                ρ = 1/inner(x, s, y)
                if ρ < 0
                    for j = 1:k
                        popfirst!(H)
                    end
                    break
                end
            end
            @inbounds H[k] = (s, y, ρ)
        end
        gprev = transport!(gprev, xprev, dprev, α, x)
        # dprev = transport!(dprev, xprev, dprev, α, x)
        dprev = dx
        y = scale!(add!(gprev, g, -1), -1)
        s = scale!(dprev, α)
        ρ = 1/inner(x, y, s)
        if ρ < 0
            empty!(H)
        else
            push!(H, (s, y, ρ))
        end
        d = let x = x
            scale!(H(g, (y1,y2)->inner(x, y1, y2), add!, scale!), -1)
        end

        verbosity >= 2 &&
            @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e",
                            numiter, f, normgrad, α)
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
    return x, f, g, normgradhistory
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
    @inbounds v = H[H.first]
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
