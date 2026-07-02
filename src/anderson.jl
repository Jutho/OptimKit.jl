"""
    struct AndersonMixing{T <: Real} <: FixedPointAlgorithm
    AndersonMixing(m::Int;
                      damping::Real = 1,
                      maxiter::Int=MAXITER[], # 1_000_000
                      gradtol::Real=GRADTOL[], # 1e-8
                      verbosity::Int=VERBOSITY[]) # 1

Anderson mixing, also known as Anderson acceleration, for fixed point problems.

WARNING: Experimental implementation – subject to change.

## Parameters
- `m::Int`: The number of previous iterates to use for Anderson extrapolation.
- `damping::Real`: The damping parameter for Anderson extrapolation; a value of 1 corresponds to no damping, while a value between 0 and 1 corresponds to under-relaxation.
- `maxiter::Int`: The maximum number of iterations.
- `gradtol::T`: The tolerance for the norm of the residual.
- `verbosity::Int`: The verbosity level of the optimization algorithm.

The verbosity level use the following scheme:
- 0: no output
- 1: only warnings upon non-convergence
- 2: convergence information at the end of the algorithm
- 3: progress information after each iteration
"""
struct AndersonMixing{T <: Real} <: FixedPointAlgorithm
    m::Int
    damping::T
    maxiter::Int
    gradtol::T
    verbosity::Int
end
function AndersonMixing(
        m::Int = 8;
        damping::Real = 1,
        maxiter::Int = MAXITER[],
        gradtol::Real = GRADTOL[],
        verbosity::Int = VERBOSITY[]
    )
    damping′, gradtol′ = promote(damping, gradtol)
    return AndersonMixing(m, damping′, maxiter, gradtol′, verbosity)
end

using LinearAlgebra
function fixedpoint(
        fp::F, x₀, alg::AndersonMixing;
        (finalize!) = _finalize!,
        shouldstop = DefaultShouldStop(alg.maxiter),
        hasconverged = DefaultHasConverged(alg.gradtol),
        retract = _retract, invretract = _invretract,
        inner = _inner, (transport!) = _transport!,
        (scale!) = _scale!, (add!) = _add!,
        isometrictransport = (transport! == _transport! && inner == _inner)
    ) where {F}

    t₀ = time()
    verbosity = alg.verbosity
    x = x₀
    fx = fp(x)
    numfp = 1
    numiter = 0
    g = invretract(x, fx) # residual, i.e. direction from x to fx, similar to F(x) - x in the vector case
    x, _, g = finalize!(x, false, g, numiter)
    innergg = inner(x, g, g)
    normg = sqrt(innergg)
    normghistory = [normg]
    verbosity >= 2 &&
        @info @sprintf("Anderson: initializing with ‖x - F(x)‖ = %.4e", normg)
    t = time() - t₀
    _hasconverged = hasconverged(x, false, g, normg)
    _shouldstop = shouldstop(x, false, g, numfp, numiter, t)

    if !(_hasconverged || _shouldstop) # first iteration is special
        xprev = x
        gprev = g
        Δxprev = deepcopy(g)
        told = t
        x, Δx = retract(x, g, 1)
        fx = fp(x)
        numfp += 1
        numiter += 1
        g = invretract(x, fx) # residual, i.e. direction from x to fx, similar to F(x) - x in the vector case
        x, _, g = finalize!(x, false, g, numiter)
        innergg = inner(x, g, g)
        normg = sqrt(innergg)
        push!(normghistory, normg)
        t = time() - t₀
        Δt = t - told
        _hasconverged = hasconverged(x, false, g, normg)
        _shouldstop = shouldstop(x, false, g, numfp, numiter, t)
    end
    TangentType = typeof(g)
    H = AndersonHistory(alg.m, TangentType[], TangentType[])
    overlap_full = zeros(typeof(innergg), alg.m, alg.m)
    rhs_full = zeros(typeof(innergg), alg.m)
    while !(_hasconverged || _shouldstop)
        verbosity >= 3 &&
            @info @sprintf("Anderson acceleration: iter %4d, Δt %s: ‖f(x) - x‖ = %.4e", numiter, format_time(Δt), normg)

        told = t
        # compute new update direction using Anderson extrapolation
        gprev = transport!(gprev, xprev, Δxprev, 1, x)
        Δg = add!(deepcopy(g), gprev, -1)
        # Δx = transport!(deepcopy(Δx), xprev, Δx, 1, x) # transport previous Anderson extrapolation step to current point

        mold = length(H)
        push!(H, (Δg, Δx))
        for k in 1:(length(H) - 1)
            (Δgₖ, Δxₖ) = H[k]
            Δgₖ = transport!(Δgₖ, xprev, Δxprev, 1, x) # transport stored residuals to current point
            Δxₖ = transport!(Δxₖ, xprev, Δxprev, 1, x)
            H[k] = (Δgₖ, Δxₖ) # update stored residuals and steps to current point
        end
        m = length(H)

        overlap = view(overlap_full, 1:m, 1:m)
        rhs = view(rhs_full, 1:m)
        if isometrictransport
            if mold == m
                @inbounds for j in 1:(m - 1)
                    for i in 1:(m - 1)
                        overlap[i, j] = overlap[i + 1, j + 1]
                    end
                end
            end
            @inbounds for i in 1:(m - 1)
                (Δgᵢ, Δxᵢ) = H[i]
                overlap[m, i] = overlap[i, m] = inner(x, Δgᵢ, Δg)
            end
            overlap[m, m] = inner(x, Δg, Δg)
        else
            @inbounds for j in 1:m
                (Δgⱼ, Δxⱼ) = H[j]
                for i in 1:(j - 1)
                    (Δgᵢ, Δxᵢ) = H[i]
                    overlap[j, i] = overlap[i, j] = inner(x, Δgᵢ, Δgⱼ)
                end
                overlap[j, j] = inner(x, Δgⱼ, Δgⱼ)
            end
        end
        @inbounds for i in 1:m
            (Δgᵢ, Δxᵢ) = H[i]
            rhs[i] = inner(x, Δgᵢ, g)
        end
        # overlapX = [inner(x, H[i][2], H[j][2]) for i in 1:m, j in 1:m]
        # overlap += sqrt(eps(one(eltype(overlap)))) * overlapX
        overlap += LinearAlgebra.tr(overlap) / size(overlap, 1) * sqrt(eps(one(eltype(overlap)))) * I
        Γ = LinearAlgebra.cholesky(overlap) \ rhs # solve least squares problem to get Anderson coefficients
        ḡ = deepcopy(g)
        Δx = scale!(deepcopy(Δx), 0)
        @inbounds for k in 1:m
            (Δgₖ, Δxₖ) = H[k]
            ḡ = add!(ḡ, Δgₖ, -Γ[k])
            Δx = add!(Δx, Δxₖ, -Γ[k]) # compute Anderson extrapolation direction
        end
        Δx = add!(Δx, ḡ, alg.damping) # add damping to Anderson extrapolation direction

        # store current quantities as previous quantities
        xprev = x
        Δxprev = Δx
        gprev = g
        _xlast[] = x # store result in global variables to debug failures
        _glast[] = g

        # take step and evaluate new point
        x, Δx = retract(x, Δx, 1)
        fx = fp(x)
        numfp += 1
        numiter += 1
        g = invretract(x, fx) # Direction from x to fx, i.e. similar to F(x) - x in the vector case
        x, _, g = finalize!(x, false, g, numiter)
        innergg = inner(x, g, g)
        normg = sqrt(innergg)
        push!(normghistory, normg)
        t = time() - t₀
        Δt = t - told
        _hasconverged = hasconverged(x, false, g, normg)
        _shouldstop = shouldstop(x, false, g, numfp, numiter, t)

        # check stopping criteria and print info
        if _hasconverged || _shouldstop
            break
        end
    end
    if _hasconverged
        verbosity >= 2 &&
            @info @sprintf("Anderson acceleration: converged after %d iterations and time %s: ‖f(x) - x‖ = %.4e", numiter, format_time(t), normg)
    else
        verbosity >= 1 &&
            @warn @sprintf("Anderson acceleration: not converged to requested tol after %d iterations and time %s: ‖f(x) - x‖ = %.4e", numiter, format_time(t), normg)
    end
    return x, g, numfp, normghistory
end

mutable struct AndersonHistory{TangentType}
    maxlength::Int
    length::Int
    first::Int
    Δresiduals::Vector{TangentType}
    Δpositions::Vector{TangentType}
    function AndersonHistory{T}(maxlength::Int, Δresiduals::Vector{T}, Δpositions::Vector{T}) where {T}
        l = length(Δresiduals)
        @assert l == length(Δpositions) "AndersonHistory: Δresiduals and Δpositions must have the same length"
        @assert l <= maxlength "AndersonHistory: initial history length cannot exceed maxlength"
        Δresiduals = resize!(copy(Δresiduals), maxlength)
        Δpositions = resize!(copy(Δpositions), maxlength)
        return new{T}(maxlength, l, 1, Δresiduals, Δpositions)
    end
end
function AndersonHistory(maxlength::Int, Δresiduals::Vector{T}, Δpositions::Vector{T}) where {T}
    return AndersonHistory{T}(maxlength, Δresiduals, Δpositions)
end

Base.length(H::AndersonHistory) = H.length

@inline function Base.getindex(H::AndersonHistory, i::Int)
    @boundscheck if i < 1 || i > H.length
        throw(BoundsError(H, i))
    end
    n = H.maxlength
    idx = H.first + i - 1
    idx = ifelse(idx > n, idx - n, idx)
    return (getindex(H.Δresiduals, idx), getindex(H.Δpositions, idx))
end

@inline function Base.setindex!(H::AndersonHistory, (Δg, Δx), i)
    @boundscheck if i < 1 || i > H.length
        throw(BoundsError(H, i))
    end
    idx = mod1(H.first + i - 1, H.maxlength)
    setindex!(H.Δresiduals, Δg, idx)
    setindex!(H.Δpositions, Δx, idx)
    return (Δg, Δx)
end

@inline function Base.push!(H::AndersonHistory, value)
    if H.length < H.maxlength
        H.length += 1
    else
        H.first = mod1(H.first + 1, H.maxlength)
    end
    @inbounds setindex!(H, value, H.length)
    return H
end
@inline function Base.pop!(H::AndersonHistory)
    @inbounds v = H[H.length]
    H.length -= 1
    return v
end
@inline function Base.popfirst!(H::AndersonHistory)
    @inbounds v = H[1]
    H.first = mod1(H.first + 1, H.maxlength)
    H.length -= 1
    return v
end

@inline function Base.empty!(H::AndersonHistory)
    H.length = 0
    H.first = 1
    return H
end
