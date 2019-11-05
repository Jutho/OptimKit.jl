module OptimKit

import LinearAlgebra
using Printf

_retract(x, d, α) = (x + α * d, d)
_inner(x, v1, v2) = v1 === v2 ? LinearAlgebra.norm(v1)^2 : LinearAlgebra.dot(v1, v2)
_transport!(v, xold, d, α, xnew) = v
_scale!(v, α) = LinearAlgebra.rmul!(v, α)
_add!(vdst, vsrc, α) = LinearAlgebra.axpy!(α, vsrc, vdst)

abstract type OptimizationAlgorithm
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
export FletcherReeves, HestenesStiefel, PolakRibierePolyak, HagerZhang, DaiYuan
export HagerZhangLineSearch

function optimtest(fg, x, d; alpha = 0:0.001:0.1, retract = _retract, inner = _inner)
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
