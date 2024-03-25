using Reexport

@reexport using BenchmarkTools

using Turing, TuringBenchmarking, ADTypes, LogDensityProblems, LogDensityProblemsAD

using StatsPlots, BenchmarkPlots, LaTeXStrings

# AD backends.
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Enzyme: Enzyme
using ReverseDiff: ReverseDiff

Enzyme.API.runtimeActivity!(true);
Enzyme.API.typeWarning!(false);

# Add Zygote.jl.
const ADBACKENDS = [
    ADTypes.AutoForwardDiff(),
    ADTypes.AutoEnzyme(),
    ADTypes.AutoReverseDiff(false),
    ADTypes.AutoReverseDiff(true),
    ADTypes.AutoZygote(),
]

# TODO: Make PR to TuringBenchmarking to add Enzyme backend.
TuringBenchmarking.backend_label(::ADTypes.AutoEnzyme) = "Enzyme"

"""
    extract_trial(suite)

Extract the trial from a suite.

Note that this assumes that the suite contains a single trial.
"""
extract_trial(suite) = last(only(BenchmarkTools.leaves(suite)))

"""
    has_mutation(model::DynamicPPL.Model)

Check if a model involves mutation.
"""
has_mutation(::DynamicPPL.Model) = false

"""
    has_fast_impl(model::DynamicPPL.Model)

Check if a model has a fast implementation.
"""
has_fast_impl(::DynamicPPL.Model) = false

"""
    scalings(example)

Return an iterator over integers representing increasing
problem sizes for the given `example`.
"""
function scalings end

"""
    filter_adbackends(model::DynamicPPL.Model, adbackends)

Filter out backends that do not support mutation.
"""
function filter_adbackends(model::DynamicPPL.Model, adbackends)
    # No need to do anything if the model does not involve mutation.
    !has_mutation(model) && return adbackends

    # Can't do `AutoReverseDiff` or `AutoZygote` with mutation.
    return filter(adbackends) do backend
        !(backend isa Union{ADTypes.AutoZygote})
    end
end

# Make it trivial to use a distribution as a log density.
LogDensityProblems.dimension(d::Distribution) = length(d.dist)
LogDensityProblems.logdensity(d::Distribution, x) = logpdf(d.dist, x)
LogDensityProblems.capabilities(::Distribution) = LogDensityProblems.LogDensityOrder{0}()

# Plotting tools
function num_to_log10_tick(n)
    labelfunc = Plots.labelfunc(:log10, Plots.backend())
    b = log10(n)
    # Treat as integer if possible.
    return Plots.texmath2unicode(LaTeXString(labelfunc(b == floor(b) ? Int(b) : b)))
end

function plot_complexity_results(
    results,
    adbackends;
    kwargs...
)
    plot_complexity_results!(plot(), results, adbackends; kwargs...)
end

function plot_complexity_results!(
    p::Plots.Plot,
    results,
    adbackends;
    normalize_by_eval=false,
    normalization_factor=1e-6,
    xscale=:log10,
    yscale=:log10,
    marker=:circle,
    legend=:topleft,
    kwargs...
)
    results_eval = results[@tagged "linked" && "evaluation"]
    results_gradient = results[@tagged "linked" && "gradient"]
    ns = sort(collect(keys(results)))

    adbackends_filtered = filter(adbackends) do adbackend
        !isempty(results_gradient[@tagged TuringBenchmarking.backend_label(adbackend)])
    end

    for adbackend in adbackends_filtered
        adbackend_label = TuringBenchmarking.backend_label(adbackend)
        times = map(ns) do n
            # Get the evaluation time.
            trial_eval_n = extract_trial(results_eval[n])
            time_eval_n = minimum(trial_eval_n).time
adbackend
            # Get the gradient time for the different backends.
            results_gradient_n = results_gradient[n]

            trial_gradient_n = extract_trial(results_gradient_n[@tagged adbackend_label])
            time_gradient = minimum(trial_gradient_n)

            if normalize_by_eval
                time_gradient.time / time_eval_n
            else
                time_gradient.time * normalization_factor
            end
        end
        plot!(p, ns, times; label=adbackend_label, xscale, yscale, marker, legend, kwargs...)
    end

    if xscale == :log10
        xticks!(map(identity, ns), map(num_to_log10_tick, ns))
    end

    # Add a tick for every log10 int between min and max.
    if yscale == :log10
        y_lb, y_ub = ylims()
        logy_lb, logy_ub = Int(ceil(log10(y_lb))), Int(floor(log10(y_ub)))
        ytick_locs = .^(10.0, logy_lb:logy_ub)
        yticks!(ytick_locs, map(num_to_log10_tick, ytick_locs))
    end

    return p
end

