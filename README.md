# OptimKit.jl

A blisfully ignorant Julia package for gradient optimization.

---

A package for gradient optimization that tries to know or assume as little as possible about your optimization problem. So far gradient descent, conjugate gradient and the L-BFGS method have been implemented, corresponding to the functions `gd`, `cg` adn `lbfgs` respectively. This will probably change to a single function `optimize` where the algorithm is specified as an argument, and specific algorithm types for the corresponding methods will be implemented.

So far, the optimization problem is specified as a function `fval, gval = fg(x)` that returns both the function value and its gradient at a given point `x`. Here, the function value is assumed to be a real number. Both `x` and the gradient can be of any type, including tuples and named tuples. As a user, you should then also specify the following functions via keyword arguments

*    `s = inner(x, g1, g2)`: compute the inner product between two gradients or similar objects at position `x`. The `x` dependence is useful for optimization on manifolds, where this function represents the metric; in particular it should be symmetric `inner(x, g1, g2) == inner(x, g2, g1)` and real-valued.
*    `retract(x, g, α)`: take a step in direction `g` (same type as gradients) starting from point `x` and with step length `α`, returns the new `x(α)` and the local direction at that position, i.e. `dx(α)/dα`.
*    `gdst = add!(gdst, gsrc, β)`: compute the equivalent of `gdst + gsrc*β`, possibly overwriting `gdst` in place, but we always use the return value
*    `g = scale!(g, β)`: compute the equivalent of `g*β`, possibly in place, but we always use the return value. This is mostly used as `scale!(g, -1)` to compute the negative gradient as part of the step direction.
*    `g = transport!(g, x, d, α)`: transport gradient `g` along the retraction of `x` in the direction `d` (same type as a gradient) with step length `α`, can be in place but the return value is used.

The `gd` algorithm only requires the first two, `cg` and `lbfgs` require all five functions. Default values are provided to make the optimization algorithms work with standard optimization problems where `x` is a vector or `Array`, i.e. they are given by
```julia
_retract(x, d, α) = (x + α * d, d)
_inner(x, v1, v2) = v1 === v2 ? LinearAlgebra.norm(v1)^2 : LinearAlgebra.dot(v1, v2)
_transport!(v, x, d, α) = v
_add!(vdst, vsrc, β) = LinearAlgebra.axpy!(β, vsrc, vdst)
_scale!(v, β) = LinearAlgebra.rmul!(v, β)
```

Finally, `lbfgs` also takes a `Bool` keyword argument `isometrictransport`, which indices whether the transport of vectors preserves their inner product, i.e. whether
```julia
inner(x, g1, g2) == inner(retract(x, d, α), transport!(g1, x, d, α), transport!(g2, x, d, α))
```

The three functions `gd`, `cg` and `lbfgs` are called as `...(fg, x₀; kwargs...)` with `x₀` an initial guess. Keyword arguments include the above function arguments (`retract`, `inner`, `add!`, `scale!`, `transport!`) as well as `maxiter` (maximum number of iterations), `tol` (requested tolerance on the norm of the gradient, i.e. convergence is obtained if `inner(x, g, g) < tol*tol` for `_, g = fg(x)`, and `verbosity`, an integer which controls the amount of information being printed.
