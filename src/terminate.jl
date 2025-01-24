@kwdef struct DefaultHasConverged{T<:Real}
    gradtol::T
end

function (d::DefaultHasConverged)(x, f, g, normgrad)
    return normgrad <= d.gradtol
end

@kwdef struct DefaultShouldStop
    maxiter::Int
end

function (d::DefaultShouldStop)(x, f, g, numfg, numiter, t)
    return numiter >= d.maxiter
end