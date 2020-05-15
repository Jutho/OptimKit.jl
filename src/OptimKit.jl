module OptimKit

import LinearAlgebra
using Printf

_retract(x, d, α) = (x + α * d, d)
_inner(x, v1, v2) = v1 === v2 ? LinearAlgebra.norm(v1)^2 : LinearAlgebra.dot(v1, v2)
_transport!(v, xold, d, α, xnew) = v
_scale!(v, α) = LinearAlgebra.rmul!(v, α)
_add!(vdst, vsrc, α) = LinearAlgebra.axpy!(α, vsrc, vdst)

_precondition(x, g) = g
_finalize!(x, f, g, numiter) = x, f, g

abstract type OptimizationAlgorithm
end

const _xlast = Ref{Any}()
const _glast = Ref{Any}()
const _dlast = Ref{Any}()

"""
    optimize(fg, x, algorithm; retract = _retract, inner = _inner,
                    transport! = _transport!, scale! = _scale!, add! = _add!,
                    isometrictransport = (transport! == _transport! && inner == _inner))

Optimize (minimize) the objective function returned as the first value of `fg`, where the second value contains the gradient, starting from a point `x` and using the algorithm `algorithm`, which is an instance of `GradientDescent`, `ConjugateGradient` or `LBFGS`.

Check the README of this package for further details on creating an algorithm instance, as well as for the meaning of the keyword arguments and their default values.
"""
function optimize end

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
function optimtest(fg, x, d = fg(x)[2]; alpha = -0.1:0.001:0.1, retract = _retract, inner = _inner)
    f0, g0 = fg(x)
    fs = Vector{typeof(f0)}(undef, length(alpha)-1)
    dfs1 = similar(fs, length(alpha) - 1)
    dfs2 = similar(fs, length(alpha) - 1)
    for i = 1:length(alpha) - 1
        a1 = alpha[i]
        a2 = alpha[i+1]
        f1, = fg(retract(x, d, a1)[1])
        f2, = fg(retract(x, d, a2)[1])
        dfs1[i] = (f2-f1)/(a2 - a1)
        xmid, dmid = retract(x, d, (a1+a2)/2)
        fmid, gmid = fg(xmid)
        fs[i] = fmid
        dfs2[i] = inner(xmid, dmid, gmid)
    end
    alphas = collect((alpha[2:end] + alpha[1:end-1])/2)
    return alphas, fs, dfs1, dfs2
end

end # module
