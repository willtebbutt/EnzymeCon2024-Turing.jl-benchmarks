# Overview

This is a quick and dirty set up for running some benchmarks on a few different models.

A "workflow" generally looks something like
```julia
using TuringEnzymeCon2024

# Needed to make Enzyme work.
using Enzyme
Enzyme.API.runtimeActivity!(true);
Enzyme.API.typeWarning!(false);

adbackends = TuringEnzymeCon2024.ADBACKENDS
adbackends = [AutoEnzyme()]

# Construct the example.
example = TuringEnzymeCon2024.AR1Example()
example = TuringEnzymeCon2024.SatelliteExample()

# Construct the benchmark suite for different "complexity" arguments.
suite = BenchmarkTools.BenchmarkGroup()
for n in TuringEnzymeCon2024.scalings(example)
    model = TuringEnzymeCon2024.make_model(example, n)
    @info "Making suite for $n"
    suite[string(n)] = TuringEnzymeCon2024.make_turing_suite(
        model;
        adbackends=TuringEnzymeCon2024.filter_adbackends(model, adbackends),
        check=true,
        error_on_failed_backend=true,
    )
end

# Run benchmark suite!
results = run(suite; seconds=60, verbose=true)

# Visualize the results.
TuringEnzymeCon2024.plot_complexity_results(
    results,
    adbackends,
    ylabel="ms",
)
```
