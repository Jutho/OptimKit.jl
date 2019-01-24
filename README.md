# OptimKit.jl

A blisfully ignorant Julia package for gradient optimization.

---

A package for gradient optimization that tries to know or assume as little as possible about your optimization problem. So far gradient descent, conjugate gradient and the L-BFGS method have been implemented. One starts by defining the algorithm that one wants to use by creating an instance `algorithm` of either
*   `GradientDescent(; params...)`
*   `ConjugateGradient(; flavor = ..., params...)`
*   `LBFGS(m::Int; params...)`
All of them take a number of parameters, namely
*   `maxiter`: number of iterations (defaults to `typemax(Int)`, so essentially unbounded)
*   `gradtol`: convergence criterion, stop when the 2-norm of the gradient is smaller than `gradtol` (default value = `gradtol = 1e-8`)
*   `linesearch`: which linesearch algorithm to be used, currently there is only one choice, namely `HagerZhangLineSearch(;...)` (see below).
*   `verbosity`: Verbosity level, the amount of information that will be printed, either `<=0` (default value) for no information, `1` for a single STDOUT output at the end of the algorithm, or `>=2` for a one-line summary after every iteration step.

Furthermore, `LBFGS` takes a single positional argument `m::Int`, the number of previous steps to take into account in the construction of the approximate (inverse) Hessian. `ConjugateGradient` has an additional keyword argument, `flavor`, which can be any of the following:
*   `HagerZhang(; η::Real = 0.4, θ::Real = 1.0)`: default
*   `HestenesStiefel()`
*   `PolakRibierePolyak()`
*   `DaiYuan()`

The `linesearch` argument currently takes the value
```julia
HagerZhangLineSearch(; δ::Real = .1, # parameter for first Wolfe condition
                        σ::Real = .9, # paremeter for second Wolfe condition
                        ϵ::Real = 1e-6, # parameter for approximate Wolfe condition, accept fluctation of ϵ on the function value
                        θ::Real = 1/2, # used in bisection
                        γ::Real = 2/3, # determines when a bisection step is performed
                        ρ::Real = 5., # expansion parameter for initial bracket interval
                        verbosity::Int = 0)
```
The linesearch has an independent `verbosity` flag to control the output of information being printed to `STDOUT`, but by default its value is equal to `verbosity-2` of the optimization algorithm. So `ConjugateGradient(; verbosity = 3)` is equivalent to
having `verbosity=1` in the linesearch algorithm.

This optimization algorithm can then be applied by calling
```julia
x, fx, gx, normgradhistory = optimize(fg, x₀, algorithm; kwargs...)
```
Here, the optimization problem (objective function) is specified as a function `fval, gval = fg(x)` that returns both the function value and its gradient at a given point `x`. The function value `fval` is assumed to be a real number of some type `T<:Real`. Both `x` and the gradient `gval` can be of any type, including tuples and named tuples. As a user, you should then also specify the following functions via keyword arguments

*    `s = inner(x, g1, g2)`: compute the inner product between two gradients or similar objects at position `x`. The `x` dependence is useful for optimization on manifolds, where this function represents the metric; in particular it should be symmetric `inner(x, g1, g2) == inner(x, g2, g1)` and real-valued.
*    `retract(x, g, α)`: take a step in direction `g` (same type as gradients) starting from point `x` and with step length `α`, returns the new `x(α)` and the local direction at that position, i.e. `dx(α)/dα`.
*    `g = scale!(g, β)`: compute the equivalent of `g*β`, possibly in place, but we always use the return value. This is mostly used as `scale!(g, -1)` to compute the negative gradient as part of the step direction.
*    `gdst = add!(gdst, gsrc, β)`: compute the equivalent of `gdst + gsrc*β`, possibly overwriting `gdst` in place, but we always use the return value
*    `g = transport!(g, x, d, α)`: transport gradient `g` along the retraction of `x` in the direction `d` (same type as a gradient) with step length `α`, can be in place but the return value is used.

The `GradientDescent` algorithm only requires the first three, `ConjugateGradient` and `LBFGS` require all five functions. Default values are provided to make the optimization algorithms work with standard optimization problems where `x` is a vector or `Array`, i.e. they are given by
```julia
_retract(x, d, α) = (x + α * d, d)
_inner(x, v1, v2) = v1 === v2 ? LinearAlgebra.norm(v1)^2 : LinearAlgebra.dot(v1, v2)
_transport!(v, x, d, α) = v
_add!(vdst, vsrc, β) = LinearAlgebra.axpy!(β, vsrc, vdst)
_scale!(v, β) = LinearAlgebra.rmul!(v, β)
```

Finally, there is one keyword argument `isometrictransport::Bool` to indicate whether the transport of vectors preserves their inner product, i.e. whether
```julia
inner(x, g1, g2) == inner(retract(x, d, α), transport!(g1, x, d, α), transport!(g2, x, d, α))
```
The default value is false, unless the default transport (`_transport!`) and inner product (`_inner`) are used.
