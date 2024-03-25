using Turing, Distributions, LinearAlgebra

# Define the model in Turing
@model function satellite_model(
    Z,
    F,
    H,
    Q,
    R,
    x0_mean,
    P0,
    ::Type{TV} = Vector{Float64},
) where {TV}
    T = length(Z)  # Number of timesteps
    state_dim = length(x0_mean)  # State dimension
    obs_dim = size(H, 1)  # Observation dimension

    # Preallocate states
    # HACK: Enzyme doesn't work with `TV` (results in `Real`) so we gotta force that.
    TV2 = eltype(TV) >: Real ? Vector{Float64} : TV
    x = Vector{TV2}(undef, T)
    x[1] ~ MvNormal(x0_mean, P0)

    for t = 2:T
        # State transition prior
        x[t] ~ MvNormal(F * x[t-1], Q)
    end

    for t = 1:T
        # Observation likelihood
        Z[t] ~ MvNormal(H * x[t], R)
    end

    return (; x, Z)
end

# This model won't work with Zygote.jl.
has_mutation(::DynamicPPL.Model{typeof(satellite_model)}) = true

# Example.
struct SatelliteExample <: AbstractExample end

scalings(::SatelliteExample) = [10, 100, 1000]

function make_model(::SatelliteExample, T)
    # Define the time step Δt
    Δt = 1.0

    # Define the system dynamics (F) and observation (H) matrices
    F = [1 0 Δt 0; 0 1 0 Δt; 0 0 1 0; 0 0 0 1]
    H = [1.0 0 0 0; 0 1 0 0]

    # Process and observation noise covariance matrices
    Q = Diagonal(diagm(0 => [0.1, 0.1, 0.1, 0.1]))
    R = Diagonal(diagm(0 => [0.1, 0.1]))

    # Initial state mean and covariance
    x0_mean = [0.0, 0.0, 1.0, 1.0]
    P0 = Diagonal(diagm(0 => [0.1, 0.1, 0.1, 0.1]))

    # Generate a circular trajectory with some radial noise.
    times = range(0, 2π, T)
    r² = 10
    r²_samples = r² .+ randn(length(times))
    r_samples = sqrt.(r²_samples)
    Z_gen = [r_samples[i] * [cos(t), sin(t)] for (i, t) in enumerate(times)]

    # Instantiate the model
    model = satellite_model(Z_gen, F, H, Q, R, x0_mean, P0)

    return model
end
