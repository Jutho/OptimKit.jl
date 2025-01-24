using VectorInterface

struct TangentVector{M,T,F1,F2,F3}
    x::M
    v::T
    inner::F1
    add!!::F2
    scale!!::F3
end

Base.getindex(tv::TangentVector) = tv.v
base(tv::TangentVector) = tv.x
function checkbase(tv1::TangentVector, tv2::TangentVector)
    return tv1.x === tv2.x ? tv1.x :
           throw(ArgumentError("tangent vectors with different base points"))
end

function VectorInterface.scale!!(tv::TangentVector, α::Real)
    tv.v = tv.scale!!(tv.v, α)
    return tv
end
function VectorInterface.add!!(tv::TV, α::Real, tv2::TV) where {TV<:TangentVector}
    checkbase(tv, tv2)
    tv.v = tv.add!!(tv.v, α, tv2.v)
    return tv
end
function VectorInterface.inner(tv::TV, tv2::TV) where {TV<:TangentVector}
    return tv.inner(checkbase(tv, tv2), tv.v, tv2.v)
end

function retract(x::M, η::TangentVector{M}, α::Real) where {M}

    return x, η
end