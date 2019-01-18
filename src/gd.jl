function gd(fg, x; linesearch = HagerZhangLineSearch(),
                    retract = _retract, inner = _inner, scale! = _scale!,
                    maxiter = typemax(Int), gradtol = 1e-8, verbosity::Int = 0)

    f, g = fg(x)
    normgrad = sqrt(inner(x, g, g))
    normgradhistory = [normgrad]
    d = scale!(deepcopy(g), -1)
    α = 1e-2
    numiter = 0
    verbosity >= 2 &&
        @info @sprintf("GD: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while numiter < maxiter
        numiter += 1
        x, f, g, α, = linesearch(fg, x, d, (f, g);
            initialguess = 1.1α, retract = retract, inner = inner, verbosity = verbosity-1)
        normgrad = sqrt(inner(x, g, g))
        push!(normgradhistory, normgrad)
        if normgrad <= gradtol
            break
        end
        verbosity >= 2 &&
            @info @sprintf("GD: iter %4d: f = %.12f, ‖∇f‖ = %.4e, step size = %.2e",
                            numiter, f, normgrad, α)
        # next search direction
        d = scale!(deepcopy(g), -1)
    end
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("GD: converged after %d iterations: f = %.12f, ‖∇f‖ = %.4e",
                            numiter, f, normgrad)
        else
            @warn @sprintf("GD: not converged to requested tol: f = %.12f, ‖∇f‖ = %.4e",
                            f, normgrad)
        end
    end
    return x, f, g, normgradhistory
end
