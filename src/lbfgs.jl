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

function optimize(fg, x, alg::LBFGS; precondition = _precondition,
                    retract = _retract, inner = _inner, transport! = _transport!,
                    scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

    verbosity = alg.verbosity
    f, g = fg(x)
    numfg = 1
    innergg = inner(x, g, g)
    normgrad = sqrt(innergg)
    normgradhistory = [normgrad]

    TangentType = typeof(g)
    ScalarType = typeof(innergg)
    m = alg.m
    H = LBFGSInverseHessian(m, TangentType[], TangentType[], ScalarType[])

    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("LBFGS: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)

    while true
        # compute new search direction
        if length(H) > 0
            Hg = let x = x
                H(g, ξ->precondition(x, ξ), (ξ1, ξ2)->inner(x, ξ1, ξ2), add!, scale!)
            end
            η = scale!(Hg, -1)
        else
            Pg = precondition(x, deepcopy(g))
            normPg = sqrt(inner(x, Pg, Pg))
            η = scale!(Pg, -1/normPg)
        end

        # store current quantities as previous quantities
        xprev = x
        gprev = g
        ηprev = η

        # perform line search
        _xlast[] = x # store result in global variables to debug linesearch failures
        _glast[] = g
        _dlast[] = η
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
            initialguess = 1., acceptfirst = true, retract = retract, inner = inner)
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
            @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, m = %d",
                            numiter, f, normgrad, α, length(H))

        # transport gprev, ηprev and vectors in Hessian approximation to x
        gprev = transport!(gprev, xprev, ηprev, α, x)
        for k = 1:length(H)
            @inbounds s, y, ρ = H[k]
            s = transport!(s, xprev, ηprev, α, x)
            y = transport!(y, xprev, ηprev, α, x)
            H[k] = (s, y, ρ)
        end
        ηprev = transport!(deepcopy(ηprev), xprev, ηprev, α, x)

        if isometrictransport
            # TRICK TO ENSURE LOCKING CONDITION IN THE CONTEXT OF LBFGS
            #-----------------------------------------------------------
            # (see A BROYDEN CLASS OF QUASI-NEWTON METHODS FOR RIEMANNIAN OPTIMIZATION)
            # define new isometric transport such that, applying it to transported ηprev,
            # it returns a vector proportional to ξ but with the norm of ηprev
            # still has norm normη because transport is isometric
            normη = sqrt(inner(x, ηprev, ηprev))
            normξ = sqrt(inner(x, ξ, ξ))

            β = normη/normξ
            ξ₁ = ηprev
            ξ₂ = scale!(ξ, β)
            ν₁ = add!(ξ₁, ξ₂, +1)
            ν₂ = scale!(deepcopy(ξ₂), -2)
            squarednormν₁ = inner(x, ν₁, ν₁)
            squarednormν₂ = inner(x, ν₂, ν₂)
            # apply Householder transforms to gprev, ηprev and vectors in H
            gprev = add!(gprev, ν₁, -2*inner(x, ν₁, gprev)/squarednormν₁)
            gprev = add!(gprev, ν₂, -2*inner(x, ν₂, gprev)/squarednormν₂)
            for k = 1:length(H)
                @inbounds s, y, ρ = H[k]
                s = add!(s, ν₁, -2*inner(x, ν₁, s)/squarednormν₁)
                s = add!(s, ν₂, -2*inner(x, ν₂, s)/squarednormν₂)
                y = add!(y, ν₁, -2*inner(x, ν₁, y)/squarednormν₁)
                y = add!(y, ν₂, -2*inner(x, ν₂, y)/squarednormν₂)
                H[k] = (s, y, ρ)
            end
            ηprev = ξ₂
        else
            # use cautious update below; see "A Riemannian BFGS Method without
            # Differentiated Retraction for Nonconvex Optimization Problems"
            β = one(normgrad)
        end

        # set up quantities for LBFGS update
        y = add!(scale!(deepcopy(g), 1/β), gprev, -1)
        s = scale!(ηprev, α)
        innersy = inner(x, s, y)
        innerss = inner(x, s, s)

        if innersy/innerss > 1e-4*normgrad
            norms = sqrt(innerss)
            ρ = innerss/innersy
            push!(H, (scale!(s, 1/norms), scale!(y, 1/norms), ρ))
        end
    end
    if verbosity > 0
        if normgrad <= alg.gradtol
            @info @sprintf("LBFGS: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e",
                            numiter, f, normgrad)
        else
            @warn @sprintf("LBFGS: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                            f, normgrad)
        end
    end
    return x, f, g, numfg, normgradhistory
end

mutable struct LBFGSInverseHessian{TangentType,ScalarType}
    maxlength::Int
    length::Int
    first::Int
    S::Vector{TangentType}
    Y::Vector{TangentType}
    ρ::Vector{ScalarType}
    α::Vector{ScalarType} # work space
    function LBFGSInverseHessian{T1,T2}(maxlength::Int, S::Vector{T1}, Y::Vector{T1}, ρ::Vector{T2}) where {T1, T2}
        @assert length(S) == length(Y) == length(ρ)
        l = length(S)
        S = resize!(copy(S), maxlength)
        Y = resize!(copy(Y), maxlength)
        ρ = resize!(copy(ρ), maxlength)
        α = similar(ρ)
        return new{T1,T2}(maxlength, l, 1, S, Y, ρ, α)
    end
end
LBFGSInverseHessian(maxlength::Int, S::Vector{T1}, Y::Vector{T1}, ρ::Vector{T2}) where {T1, T2} = LBFGSInverseHessian{T1, T2}(maxlength, S, Y, ρ)

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

function (H::LBFGSInverseHessian)(g, precondition, inner, add!, scale!; α = H.α)
    q = deepcopy(g)
    for k = length(H):-1:1
        s, y, ρ = H[k]
        α[k] = ρ * inner(s, q)
        q = add!(q, y, -α[k])
    end
    s, y, ρ = H[length(H)]
    γ = inner(s, y)/inner(y, precondition(y))
    z = scale!(precondition(q), γ)
    for k = 1:length(H)
        s, y, ρ = H[k]
        β = ρ * inner(y, z)
        z = add!(z, s, (α[k]-β))
    end
    return z
end
