module OptimKit

using LinearAlgebra: LinearAlgebra
using Printf
using ScopedValues
using VectorInterface
using Base: @kwdef

# Default values for the keyword arguments using ScopedValues
const LS_MAXITER = ScopedValue(10)
const LS_MAXFG = ScopedValue(20)
const LS_VERBOSITY = ScopedValue(1)

const GRADTOL = ScopedValue(1e-8)
const MAXITER = ScopedValue(1_000_000)
const VERBOSITY = ScopedValue(1)

# Default values for the manifold structure
_retract(x, d, α) = (add(x, d, α), d)
_inner(x, v1, v2) = v1 === v2 ? norm(v1)^2 : real(inner(v1, v2))
_transport!(v, xold, d, α, xnew) = v
_scale!(v, α) = scale!!(v, α)
_add!(vdst, vsrc, α) = add!!(vdst, vsrc, α)

_precondition(x, g) = deepcopy(g)
_finalize!(x, f, g, numiter) = x, f, g

# Default structs for new convergence and termination keywords
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

# Optimization
abstract type OptimizationAlgorithm end

const _xlast = Ref{Any}()
const _glast = Ref{Any}()
const _dlast = Ref{Any}()

"""
    optimize(fg, x, alg;
                  precondition=_precondition,
                  (finalize!)=_finalize!,
                  hasconverged=DefaultHasConverged(alg.gradtol),
                  shouldstop=DefaultShouldStop(alg.maxiter),
                  retract=_retract, inner=_inner, (transport!)=_transport!,
                  (scale!)=_scale!, (add!)=_add!,
                  isometrictransport=(transport! == _transport! && inner == _inner))
    -> x, f, g, numfg, history

Optimize (minimize) the objective function returned as the first value of `fg`, where the
second value contains the gradient, starting from a point `x` and using the algorithm
`algorithm`, which is an instance of `GradientDescent`, `ConjugateGradient` or `LBFGS`.

Returns the final point `x`, the coresponding function value `f` and gradient `g`, the
total number of calls to `fg`, and the history of the gradient norm across the different
iterations.

The algorithm is run until either `hasconverged(x, f, g, norm(g))` returns `true` or
`shouldstop(x, f, g, numfg, numiter, time)` returns `true`. The latter case happening before
the former is considered to be a failure to converge, and a warning is issued.

The keyword arguments are:

-   `precondition::Function`: A function that takes the current point `x` and the gradient `g`
    and returns a preconditioned gradient. By default, the identity is used.
-   `finalize!::Function`: A function that takes the final point `x`, the function value `f`,
    the gradient `g`, and the iteration number, and returns a possibly modified values for
    `x`, `f` and `g`. By default, the identity is used.
    It is the user's responsibility to ensure that the modified values do not lead to
    inconsistencies within the optimization algorithm.
-   `hasconverged::Function`: A function that takes the current point `x`, the function value `f`,
    the gradient `g`, and the norm of the gradient, and returns a boolean indicating whether
    the optimization has converged. By default, the norm of the gradient is compared to the
    tolerance `gradtol` as encoded in the algorithm instance.
-   `shouldstop::Function`: A function that takes the current point `x`, the function value `f`,
    the gradient `g`, the number of calls to `fg`, the iteration number, and the time spent
    so far, and returns a boolean indicating whether the optimization should stop. By default,
    the number of iterations is compared to the maximum number of iterations as encoded in the
    algorithm instance.

Check the README of this package for further details on creating an algorithm instance,
as well as for the meaning of the remaining keyword arguments and their default values.

!!! Warning
    
    The default values of `hasconverged` and `shouldstop` are provided to ensure continuity
    with the previous versions of this package. However, this behaviour might change in the
    future.

Also see [`GradientDescent`](@ref), [`ConjugateGradient`](@ref), [`LBFGS`](@ref).
"""
function optimize end

function format_time(t::Float64)
    return t < 60 ? @sprintf("%.2f s", t) :
           t < 2600 ? @sprintf("%.2f m", t / 60) :
           @sprintf("%.2f h", t / 3600)
end

include("linesearches.jl")
include("gd.jl")
include("cg.jl")
include("lbfgs.jl")

const gd = GradientDescent()
const cg = ConjugateGradient()
const lbfgs = LBFGS()

export optimize, gd, cg, lbfgs, optimtest
export GradientDescent, ConjugateGradient, LBFGS
export FletcherReeves, HestenesStiefel, PolakRibiere, HagerZhang, DaiYuan
export HagerZhangLineSearch

"""
    optimtest(fg, x, [d]; alpha = -0.1:0.001:0.1, retract = _retract, inner = _inner)
    -> αs, fs, dfs1, dfs2

Test the compatibility between the computation of the gradient, the retraction and the inner product by computing the derivative of the objective function along a curve corresponding to a retraction starting at the point `x` in the direction `d` in two different ways. In particular, at point `αs` which are in the middle of the points in the original range or list `alpha`, both the function value `fs` as well as the two different values for the derivative `dfs1` and `dfs2` are returned, where the derivatives are computed by
1.  numerical differentation, i.e. `dfs1` contains the values `(fg(retract(x, d, alpha[i+1])[1])[1] - fg(retract(x, d, alpha[i])[1])[1])/(alpha[i+1]-alpha[i])` as an estimate for the derivative at the point `(alpha[i]+alpha[i+1])/2`
2.  using the gradient, i.e. `dfs2` contains the values `inner(xα, dα, gα)` where `xα, dα = retract(x, d, α)` for values `α = (alpha[i]+alpha[i+1])/2` and `gα = fg(xα)[2]`.

It is up to the user to check that the values in `dfs1` and `dfs2` match up to expected precision, by inspecting the numerical values or plotting them. If these values don't match, the linesearch in `optimize` cannot be expected to work.
"""
function optimtest(fg, x, d=fg(x)[2]; alpha=-0.1:0.001:0.1, retract=_retract, inner=_inner)
    f0, g0 = fg(x)
    fs = Vector{typeof(f0)}(undef, length(alpha) - 1)
    dfs1 = similar(fs, length(alpha) - 1)
    dfs2 = similar(fs, length(alpha) - 1)
    for i in 1:(length(alpha) - 1)
        a1 = alpha[i]
        a2 = alpha[i + 1]
        f1, = fg(retract(x, d, a1)[1])
        f2, = fg(retract(x, d, a2)[1])
        dfs1[i] = (f2 - f1) / (a2 - a1)
        xmid, dmid = retract(x, d, (a1 + a2) / 2)
        fmid, gmid = fg(xmid)
        fs[i] = fmid
        dfs2[i] = inner(xmid, dmid, gmid)
    end
    alphas = collect((alpha[2:end] + alpha[1:(end - 1)]) / 2)
    return alphas, fs, dfs1, dfs2
end

end # module
