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

export optimize, gd, cg, lbfgs
export GradientDescent, ConjugateGradient, LBFGS
export FletcherReeves, HestenesStiefel, PolakRibierePolyak, HagerZhang, DaiYuan
export HagerZhangLineSearch

end # module
