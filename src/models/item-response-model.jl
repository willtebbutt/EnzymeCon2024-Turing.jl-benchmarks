using Turing, StatsFuns

# TODO: Add some mechanism for scaling the complexity of the model.

### Setup ###
function sim(I, P)
    yvec = Vector{Int}(undef, I * P)
    ivec = similar(yvec)
    pvec = similar(yvec)

    beta = rand(Normal(), I)
    theta = rand(Normal(), P)

    n = 0
    for i = 1:I, p = 1:P
        n += 1
        ivec[n] = i
        pvec[n] = p
        yvec[n] = rand(BernoulliLogit(theta[p] - beta[i]))
    end

    return yvec, ivec, pvec, theta, beta
end

### Turing ###
# naive implementation
@model function irt_naive(y, i, p; I = maximum(i), P = maximum(p))
    theta ~ filldist(Normal(), P)
    beta ~ filldist(Normal(), I)

    for n in eachindex(y)
        y[n] ~ BernoulliLogit(theta[p[n]] - beta[i[n]])
    end
end

has_fast_impl(::DynamicPPL.Model{typeof(irt_naive)}) = true

# performant model
function bernoulli_logit_logpdf(y, theta, beta)
    return logpdf(BernoulliLogit(theta - beta), y)
end

@model function irt(y, i, p; I = maximum(i), P = maximum(p))
    theta ~ filldist(Normal(), P)
    beta ~ filldist(Normal(), I)
    Turing.@addlogprob! sum(bernoulli_logit_logpdf.(y, theta[p], beta[i]))

    return (; theta, beta)
end

# Example.
struct IRTExample <: AbstractExample end

scalings(::IRTExample) = [10, 100, 1000, 10000]

function make_model(::IRTExample, P, I = 20)
    y, i, p, _, _ = sim(I, P)
    return irt_naive(y, i, p)
end

function make_fast_model(::IRTExample, P, I = 20)
    y, i, p, _, _ = sim(I, P)
    return irt(y, i, p)
end
