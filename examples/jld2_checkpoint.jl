"""
Demo: checkpointing and resuming LBFGS optimization with JLD2.

Usage (from the repo root):
    julia --project=. test/jld2_checkpoint_demo.jl

What it shows:
  1. Run LBFGS with a `checkpoint` callback that saves state to a JLD2 file
     every N iterations.
  2. Interrupt early (via `shouldstop`) to simulate a crashed job.
  3. Load the last checkpoint from disk and resume to full convergence.
  4. Verify the resumed result matches a reference run.
"""

using OptimKit
using LinearAlgebra
using JLD2          # `import Pkg; Pkg.add("JLD2")` if not yet installed

# ---------------------------------------------------------------------------
# Problem: minimise  f(x) = ½ (x-y)ᵀ A (x-y)
# ---------------------------------------------------------------------------
function make_fg(A, y)
    function fg(x)
        r = x - y
        g = A * r
        f = dot(r, g) / 2
        return f, g
    end
    return fg
end

# Reproducible random problem
import Random; Random.seed!(42)
n  = 50
y  = randn(n)
A  = let B = randn(n, n); B'B + 5I end   # positive-definite, well-conditioned
fg = make_fg(A, y)
x₀ = randn(n)
alg = LBFGS(; gradtol=1e-12, verbosity=0)

# ---------------------------------------------------------------------------
# Reference: run to convergence (ground truth)
# ---------------------------------------------------------------------------
x_ref, f_ref, _, _, _ = optimize(fg, x₀, alg)
println("Reference: f* = $f_ref,  ‖x*-y‖ = $(norm(x_ref - y))")

# ---------------------------------------------------------------------------
# Helper: build a checkpoint callback that saves to `filepath` every
# `save_every` completed iterations using JLD2.
# ---------------------------------------------------------------------------
function make_jld2_checkpoint(filepath::String; save_every::Int=1)
    function checkpoint(state::LBFGSState)
        if mod(state.numiter, save_every) == 0
            jldsave(filepath; state)
            # Uncomment the line below to see checkpoint progress:
            # println("  [checkpoint] saved iter $(state.numiter), f=$(state.f)")
        end
    end
    return checkpoint
end

# ---------------------------------------------------------------------------
# Phase 1: run for up to 10 iterations, saving a checkpoint after each one
# ---------------------------------------------------------------------------
checkpoint_file = tempname() * ".jld2"

checkpoint_cb = make_jld2_checkpoint(checkpoint_file; save_every=1)
stop_at_10     = (x, f, g, numfg, numiter, t) -> numiter >= 10

x_part, f_part, _, numfg_part, history_part =
    optimize(fg, x₀, alg;
             checkpoint  = checkpoint_cb,
             shouldstop  = stop_at_10,
             hasconverged = (x, f, g, ng) -> ng <= 1e-12)

println("\nPhase 1 done: $(size(history_part,1)-1) iterations, f = $f_part")
println("Checkpoint file: $checkpoint_file  ($(round(filesize(checkpoint_file)/1024, digits=1)) KB)")

# ---------------------------------------------------------------------------
# Phase 2: load checkpoint and resume to convergence
# ---------------------------------------------------------------------------
state_loaded = jldopen(checkpoint_file, "r") do file
    file["state"]
end

println("\nLoaded checkpoint: numiter=$(state_loaded.numiter), numfg=$(state_loaded.numfg)")
println("  fhistory length = $(length(state_loaded.fhistory))  (should be numiter+1)")
println("  H length        = $(length(state_loaded.H))         (LBFGS memory used)")

x_resumed, f_resumed, _, numfg_resumed, history_resumed =
    optimize(fg, state_loaded, alg)

println("\nPhase 2 done: total $(size(history_resumed,1)-1) iterations, f = $f_resumed")
println("  numfg (total)   = $numfg_resumed")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
@assert x_resumed ≈ x_ref rtol=1e-8  "resumed solution differs from reference"
@assert f_resumed ≈ f_ref rtol=1e-8  "resumed f* differs from reference"
@assert history_resumed[1:size(history_part,1), :] ≈ history_part "history mismatch"

println("\n✓ All checks passed — resumed result matches reference run.")

# Clean up temp file
rm(checkpoint_file)
