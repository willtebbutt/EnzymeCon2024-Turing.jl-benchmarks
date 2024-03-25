# Overview

This is a quick and dirty set up for running some benchmarks on a few different models.

A "workflow" generally looks something like
```julia
# Include the setup file.
include("setup.jl")

# Include the examples.
include("models.jl")

# Construct the example.
example = SatelliteExample()

# Construct the benchmark suite for different "complexity" arguments.
suite = BenchmarkTools.BenchmarkGroup()
for n in scalings(example)
    model = make_model(example, n)
    @info "Making suite for $n"
    suite[n] = make_turing_suite(
        model;
        adbackends=filter_adbackends(model, adbackends),
        check=true,
        error_on_failed_backend=true,
    )
end

# Run benchmark suite!
results = run(suite; seconds=1, verbose=true)

# Visualize the results.
plot_complexity_results(
    results,
    adbackends,
    ylabel="ms",
)
```
