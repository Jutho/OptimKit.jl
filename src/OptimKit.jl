module OptimKit

import LinearAlgebra
using Printf

_retract(x, d, α) = (x + α * d, d)
_inner(x, v1, v2) = v1 === v2 ? LinearAlgebra.norm(v1)^2 : LinearAlgebra.dot(v1, v2)
_transport(v, x, d, α) = v
_scale!(v, α) = LinearAlgebra.rmul!(v, α)
_add!(vdst, vsrc, α) = LinearAlgebra.axpy!(α, vsrc, vdst)

include("linesearches.jl")
include("gd.jl")
include("cg.jl")
include("lbfgs.jl")

export gd, cg, lbfgs
export FletcherReeves, HestenesStiefel, PolakRibierePolyak, HagherZhang, DaiYuan
export HagerZhangLineSearch

end # module
