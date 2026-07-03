# OptimKit.jl

A blissfully ignorant Julia package for optimization and fixed-point iteration.

OptimKit is designed to make as few assumptions as possible about your state type and
geometry. It supports both classic (Euclidean) vector problems, as well as problems where
the configuration space has the structure of a Riemannian manifold, through user-provided
`retract`, `transport!`, `inner`, and related callbacks.

## Optimization API

The main functionality of this package is to mimize a cost function `f` using gradient-based
optimization techniques.

```julia
x, fx, gx, numfg, history = optimize(fg, x0, algorithm; kwargs...)
```

Here, the optimization problem (objective function) is specified as a function
`fval, gval = fg(x)` that returns both the function value and its gradient at a given point
`x`. The function value `fval` is assumed to be a real number of some type `T<:Real`. Both
`x` and the gradient `gval` can be of any type, including tuples and named tuples. The
keyword arguments to `optimize` specify how to deal with such general points and gradients,
as discussed in the first subsection below. The third positional argument specifies which
algorithm to use, and its possible values are discussed in the "Optimization algorithms"
subsection below.

The return value consists of

- `x`: final iterate
- `fx`, `gx`: cost function value and gradient at the final iterate, i.e. `fx, gx = fg(x)`.
- `numfg`: number of calls to `fg`
- `history`: function values and gradient norms over the different iterations, returned as
  matrix of size `(numiter, 2)`

### Keyword arguments related to the geometry

The first list of keyword arguments is given by `retract`, `inner`, `scale!`, `add!` and
`transport!`. Their values should be functions that specify the manifold structure and of
the configuration space and its tangent space, with the following signature and behavior:

*    `x, ξ = retract(x₀, η, α)`: take a step in direction `η` (same type as gradients)
     starting from point `x₀` and with step length `α`, returns the new
     ``x(α) = Rₓ₀(α * η)`` and the local tangent to this path at that position, i.e.
     ``ξ = D Rₓ₀(α * η)[η]`` (informally, ``ξ = dx(α) / dα``).
*    `s = inner(x, ξ1, ξ2)`: compute the inner product between two gradients or similar
     objects at position `x`. The `x` dependence is useful for optimization on manifolds,
     where this function represents the Riemannian metric; in particular it should be
     symmetric `inner(x, ξ1, ξ2) == inner(x, ξ2, ξ1)` and real-valued.
*    `η = scale!(η, β)`: compute the equivalent of `η*β`, possibly in place, but we always
     use the return value. This is mostly used as `scale!(g, -1)` to compute the negative
     gradient as part of the step direction.
*    `η = add!(η, ξ, β)`: compute the equivalent of `η + ξ*β`, possibly overwriting `η` in
     place, but we always use the return value
*    `ξ = transport!(ξ, x, η, α, x′)`: transport tangent vector `ξ` along the retraction of
     `x` in the direction `η` (same type as a gradient) with step length `α`, can be in
     place but the return value is used. Transport also receives `x′ = retract(x, η, α)[1]`
     as final argument, which has been computed before and can contain useful data that does
     not need to be recomputed

The default values for these keyword arguments assume the case of a real Euclidean vector
space (and the absense of a preconditioner), and are specified using methods from
[VectorInterface.jl](https://github.com/QuantumKitHub/VectorInterface.jl).

```julia
retract(x, η, α) = (add(x, d, α), d)
inner(x, ξ1, ξ2) = ξ1 === ξ2 ? LinearAlgebra.norm(ξ1)^2 : real(VectorInterface.inner(ξ1, ξ2))
transport!(ξ, x, η, α, xnew) = ξ
add!(η, ξ, β) = VectorInterface.add!!(η, ξ, β)
scale!(η, β) = VectorInterface.scale!!(η, β)
```

These default values enable the use of real- or complex-valued arrays, as well as nested
structures involving tuples and arrays, as supported by VectorInterface.jl, under the
assumption of a Euclidean vector space structure.

To change the geometrical structure or enable the use of other data types, custum
implementations for these functions need to be provided through the keyword arguments, and
correctness thereof should be ensured by the user. In particular, note that the gradient `g`
of the objective function should satisfy ``d f(x(α)) / d α  = inner(x(α), ξ(α), g(x(α)))``.
There is a utility function `optimtest` to facilitate testing this compatibility relation
for your given choice of `fg`, `retract` and `inner`.

There is one more keyword argument related to the geometry, namely
`isometrictransport::Bool`, which is used to indicate whether the transport of vectors
preserves their inner product, i.e. whether
```julia
inner(x, ξ1, ξ2) == inner(retract(x, η, α), transport!(ξ1, x, η, α), transport!(ξ2, x, η, α))
```
The default value is false, unless the default transport and inner product are used. When
the inner product is preserved, this can reduce the number of calls to `inner`, and can also
be exploited in some of the optimization algorithms (see below).

### Additional keyword arguments

The `optimize` function accepts more keyword arguments that can be used to control the
optimization process, namely `precondition`, `finalize!`, `hasconverged` and `shouldstop`.
These too should have a function as value with the following signature and behavior:

*    `Pη = precondition(x, η)`: apply a preconditioner to the current gradient or tangent
     vector `η` at the position `x`; the resulting `Pη` is assumed to not share memory with
     `η` even if a trivial preconditioner is applied.
*    `x, f, g = finalize!(x, f, g, numiter)`: after every step (i.e. upon completion of the
     linesearch), allows to modify the position and corresponding function value or
     gradient, or to do other tasks such as printing out statistics. Note that this step
     happens before computing new directions in Conjugate Gradient and LBFGS, so if `f` and
     `g` are modified, this is at the user's own risk (e.g. Wolfe conditions might no longer
     be satisfied, ...).
*    `bool = hasconverged(x, f, g, normg)`: a function that decides whether the optimization
     has converged and can thus be stopped (return value `true`) based on the point `x`, the
     function value `f`, the gradient `g` and the gradient norm `normg`.
*   `bool = shouldstop(x, f, g, numfg, numiter, t_elapsed)`: a function that decides whether
    the optimization should halt without converging (return value `true`) based on the point
    `x`, the function value `f`, the gradient `g`, the total number `numfg` of `fg` calls,
    the total number of iterations `numiter` in the optimization routine, and the total time
    `t_elapsed` since entering the `optimize` call.

Default values are given by `precondition(x, η) = deepcopy(η)`,
`finalize!(x, f, g, numiter) = x, f, g`,
`hasconverged(x, f, g, normg) = normg < algorithm.gradtol` and
`shouldstop(x, f, g, numfg, numiter, t_elapsed) = numiter > algorithm.maxiter`, where the algorithm
fields are explained in the next subsection.


### Optimization algorithms

The third positional argument to `optimize` can take on of the following values

- `GradientDescent(; kwargs...)`
- `ConjugateGradient(; flavor = ..., kwargs...)`
- `LBFGS(m::Int; kwargs...)`

All three algorithms share the following keyword arguments:

- `maxiter`: maximum number of iterations (default effectively unbounded)
- `gradtol`: stop when `norm(g)` drops below this threshold (default `1e-8`)
- `verbosity`: controls logging verbosity
- `linesearch`: line search algorithm (currently only `HagerZhangLineSearch`, see below)

`LBFGS` also takes:

- positional argument `m::Int`: history size for inverse-Hessian approximation
- `acceptfirst::Bool = true`: whether the first line-search trial step may be accepted

`ConjugateGradient` also takes:

- `flavor = HagerZhang(; η = 0.4, θ = 1.0)` (default)
- `flavor = HestenesStiefel()`
- `flavor = PolakRibiere()`
- `flavor = DaiYuan()`

Note that `maxiter` and `gradtol` will be ignored if the values to the keyword arguments
`hasconverged` and `shouldstop` of the `optimize` call are changed away from their default
values.

In the case of `isometrictransport == true`, the convergence of conjugate gradient and LBFGS
is more robust (and can be theoretically proven). Note that isometric transport might not be
the same as retraction transport. In particular, for `x′, ξ = retract(x, η, α)`, it is
possible that ``ξ != transport(η, x, η, α, x′)``. However, when isometric transport is
provided, we complement it with an isometric rotation such that ``ξ = D Rₓ₀(α * η)[η]`` and
``transport(η, x, η, α)`` are parallel in the case of `LBFGS`. This is the so-called locking
condition of [Huang, Gallivan and Absil](https://doi.org/10.1137/140955483), and the
approach is described in section 4.1.

### Line search

The `linesearch` keyword argument to the optimization algorithms takes the value.

```julia
HagerZhangLineSearch(; δ::Real = .1,
                       σ::Real = .9,
                       ϵ::Real = 1e-6,
                       θ::Real = 1/2,
                       γ::Real = 2/3,
                       ρ::Real = 5.,
                       verbosity::Int = 0)
```

The line search has its own `verbosity`. By default it tracks the optimization algorithm
verbosity as `verbosity - 2`.

## Fixed-Point API

OptimKit also provides a general fixed-point driver:

```julia
x, g, numfp, history = fixedpoint(fp, x0, algorithm; kwargs...)
```

where `fp(x)` returns the next iterate candidate, corresponding to the iteration process
`xₙ₊₁ = fp(xₙ)`. Here, `x` can be of any possible type, but operations on `x` are controlled
via the keyword arguments, discussed below. The third positional argument `algorithm` is
discussed in the "Algorithms" subsection below.

The return value consists of

- `x`: final iterate
- `g`: residual at the final iterate, defined as `g = invretract(x, fp(x))` (see below)
- `numfp`: number of calls to `fp`
- `history`: residual norms over iterations, returned as a vector length `numiter`.

### Keyword arguments related to the geometry

The first list of keyword arguments is given by `retract`, `invretract`, `inner`, `scale!`,
`add!`, `transport!`, `isometrictransport`. Their values should be functions that specify
the manifold structure and of the configuration space and its tangent space. Their
signature, behavior and default values are exactly the same as discussed above in the case
of `optimize`, except for `invretract` which was not needed for `optimize`, and acts as

*   `η = invretract(x₀, x)` returns the tangent vector `η` such that
    `retract(x₀, η, 1) == x`, i.e. it quantifies the separation between two points `x` and
    `x₀` on the manifold in terms of a tangent vector at base point `x₀`.

    The default value of `invretract` assumes that points live in a Euclidean vector space
    and corresponds to `invretract(x₀, x) = VectorInterface.add!!(x, x₀, -1)`.

### Additional keyword arguments

The `fixedpoint` function accepts more keyword arguments that can be used to control the
fixed-point iteration process, namely `finalize!`, `hasconverged` and `shouldstop`. They
have the same signature, behavior and default values as discussed for the case of `optimize`
above. As there is no cost function in the context of `fixedpoint`, a sentinel value
`f = false` is passed instead, whereas the residual `g = invretract(x, fp(x))` and its norm
are passed instead of the gradient of the cost function.

### Fixed-point algorithms

The third positional argument to `optimize` can take on of the following values
- `SimpleIteration(; maxiter = ..., gradtol = ..., verbosity = ...)`
- `AndersonMixing(m::Int = 8; damping = 1, maxiter = ..., gradtol = ..., verbosity = ...)`

`AndersonMixing` is experimental and may change.
