using Turing

# AR1 model.
# `truncated(Normal(), lower=0)`
@model function ar1(y)
    n = length(y)
    σ ~ InverseGamma(2, 3)
    ρ ~ Uniform(-1, 1)
    μ ~ Normal(0, 10)


    # y[i] = ...
    y[1] ~ Normal(μ, σ / sqrt(1 - ρ^2))
    for i in 2:n
        y[i] ~ Normal(μ + ρ * (y[i - 1] - μ), σ)
    end

    return y
end

has_mutation(::DynamicPPL.Model{typeof(ar1)}) = false

function make_model(n)
    model_gen = ar1(fill(missing, n))
    y_obs = model_gen()

    model = ar1(map(identity, y_obs))
    return model
end

# Example.
struct AR1Example end

scalings(::AR1Example) = [1, 10, 100, 1000]

function make_model(::AR1Example, n)
    model_gen = ar1(fill(missing, n))
    y_obs = model_gen()

    model = ar1(map(identity, y_obs))
    return model
end
