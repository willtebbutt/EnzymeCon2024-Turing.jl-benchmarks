using Turing

# HACK: Was wondering if circumventing the `istrans(vi)` call in `DynamicPPL.maybe_invlink_before_eval!!` would work,
# but if I do this, then somehow we end up with the following stacktrace:
#
# [836939] signal (11.128): Segmentation fault
# in expression starting at REPL[23]:2
# ThreadSafeVarInfo at /home/tor/.julia/packages/DynamicPPL/pg70d/src/threadsafe.jl:12
# unknown function (ip: 0x7be6b366c07f)
# macro expansion at /home/tor/.julia/packages/Enzyme/l4FS0/src/compiler.jl:5378 [inlined]
# enzyme_call at /home/tor/.julia/packages/Enzyme/l4FS0/src/compiler.jl:5056 [inlined]
# CombinedAdjointThunk at /home/tor/.julia/packages/Enzyme/l4FS0/src/compiler.jl:4998 [inlined]
# autodiff at /home/tor/.julia/packages/Enzyme/l4FS0/src/Enzyme.jl:215 [inlined]
#
# which seems even more strange (we somehow hit `ThreadSafeVarInfo` even though we were running with a single thread).

# function DynamicPPL.maybe_invlink_before_eval!!(vi::DynamicPPL.VarInfo, context::DynamicPPL.AbstractContext, model::DynamicPPL.Model)
#     # Because `VarInfo` does not contain any information about what the transformation
#     # other than whether or not it has actually been transformed, the best we can do
#     # is just assume that `default_transformation` is the correct one if `istrans(vi)`.
#     t = (DynamicPPL.transformation(vi) isa DynamicPPL.DynamicTransformation) ? DynamicPPL.NoTransformation() : DynamicPPL.transformation(vi)
#     return DynamicPPL.maybe_invlink_before_eval!!(t, vi, context, model)
# end

# Define the model.
@model truncated_normal() = x ~ truncated(Normal(), lower=0)
# Instantiate the model.
model = truncated_normal();
# Construct the log density function.
f = Turing.LogDensityFunction(model);
# NOTE: It works with `SimpleVarInfo`.
# f = Turing.LogDensityFunction(model, Turing.SimpleVarInfo(model));
# With ForwardDiff.jl to compute the gradient.
f_with_grad = ADgradient(AutoForwardDiff(chunksize=1), f);
# (✓) Works!
LogDensityProblems.logdensity_and_gradient(f_with_grad, Turing.VarInfo(model)[:])
# With Enzyme.jl to compute the gradient.
f_with_grad = ADgradient(AutoEnzyme(), f);
# (×) Breaks!
LogDensityProblems.logdensity_and_gradient(f_with_grad, Turing.VarInfo(model)[:])
