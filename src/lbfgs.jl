function lbfgs(fg, x; linesearch = HagerZhangLineSearch(),
                    retract = _retract, inner = _inner, transport = _transport,
                    scale! = _scale!, add! = _add!,
                    isometrictransport = (transport == _transport && inner == _inner),
                    m = 10, maxiter = typemax(Int), gradtol = 1e-8, verbosity::Int = 0)

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

    x, f, g, α, = linesearch(fg, x, d, (f, g);
        initialguess = 1e-2, retract = retract, inner = inner, verbosity = verbosity-1)
    normgrad = sqrt(inner(x, g, g))
    push!(normgradhistory, normgrad)

    # set up InverseHessian approximation
    gprev = transport(gprev, xprev, dprev, α)
    dprev = transport(dprev, xprev, dprev, α)
    y = scale!(add!(gprev, g, -1), -1)
    s = scale!(dprev, α)
    ρ = 1/inner(x, s, y)
    H = LBFGSInverseHessian(m,1,1, sizehint!([s], m), sizehint!([y], m), sizehint!([ρ], m))
    d = let x = x
        scale!(H(g, (y1,y2)->inner(x, y1, y2), add!, scale!), -1)
    end

    verbosity >= 2 &&
        @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e",
                        numiter, f, normgrad, α)

    while numiter < maxiter && normgrad > gradtol
        numiter += 1
        xprev = x
        dprev = d
        gprev = g

        x, f, g, α, = linesearch(fg, x, d, (f, g);
            initialguess = 1., retract = retract, inner = inner, verbosity = verbosity-1)
        normgrad = sqrt(inner(x, g, g))
        push!(normgradhistory, normgrad)
        if normgrad <= gradtol
            break
        end
        # next search direction
        for k = 1:length(H)
            @inbounds s, y, ρ = H[k]
            s = transport(s, xprev, dprev, α)
            y = transport(y, xprev, dprev, α)
            if !isometrictransport
                ρ = 1/inner(x, s, y)
                @assert ρ > 0
            end
            @inbounds H[k] = (s, y, ρ)
        end
        gprev = transport(gprev, xprev, dprev, α)
        dprev = transport(dprev, xprev, dprev, α)
        y = scale!(add!(gprev, g, -1), -1)
        s = scale!(dprev, α)
        ρ = 1/inner(x, y, s)
        @assert ρ > 0
        push!(H, (s, y, ρ))
        d = let x = x
            scale!(H(g, (y1,y2)->inner(x, y1, y2), add!, scale!), -1)
        end

        verbosity >= 2 &&
            @info @sprintf("LBFGS: iter %4d: f = %.12f, ‖∇f‖ = %.4e, α = %.2e",
                            numiter, f, normgrad, α)
    end
    if verbosity > 0
        if normgrad <= gradtol
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
end

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
        push!(H.S, s)
        push!(H.Y, y)
        push!(H.ρ, ρ)
        H.length += 1
    else
        H.first = (H.first == H.maxlength ? 1 : H.first + 1)
        @inbounds setindex!(H, (s, y, ρ), H.length)
    end
    return H
end

function (H::LBFGSInverseHessian)(g, inner, add!, scale!; α = similar(H.ρ))
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
