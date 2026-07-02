module Sphere
using Test
using VectorInterface
function retract(x, g, α)
    @test norm(x) ≈ 1
    @test abs(inner(x, g)) < sqrt(eps(eltype(x)))
    gnorm = norm(g)
    ug = g / gnorm
    θ = α * gnorm
    xnew = cos(θ) * x + sin(θ) * ug
    gnew = -sin(θ) * gnorm * x + cos(θ) * g
    gnew = add!(gnew, xnew, -inner(xnew, gnew))
    @test norm(xnew) ≈ 1
    @test abs(inner(xnew, gnew)) < sqrt(eps(eltype(x)))
    return xnew, gnew
end
function invretract(x₀, x₁)
    @test norm(x₀) ≈ 1
    @test norm(x₁) ≈ 1
    cosθ = inner(x₀, x₁)
    θ = acos(clamp(cosθ, -1, 1))
    g = (x₁ - cosθ * x₀) / sinc(θ / pi)
    return g
end
function transport(v, x₀, g₀, α, x₁)
    @test norm(x₀) ≈ 1
    @test norm(x₁) ≈ 1
    @test abs(inner(x₀, v)) < sqrt(eps(eltype(x₀)))
    @test abs(inner(x₀, g₀)) < sqrt(eps(eltype(x₀)))
    @test retract(x₀, g₀, α)[1] ≈ x₁
    gnorm = norm(g₀)
    θ = α * gnorm
    ug = g₀ / gnorm
    gv = inner(ug, v)
    v₁ = v - gv * ug
    vnew = gv * (-sin(θ) * x₀ + cos(θ) * ug) + v₁
    @test abs(inner(x₁, vnew)) < sqrt(eps(eltype(x₀)))
    @test norm(vnew) ≈ norm(v)
    vnew = add!(vnew, x₁, -inner(x₁, vnew))
    return vnew
end
end
