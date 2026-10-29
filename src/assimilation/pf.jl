"""
$(TYPEDEF)

Bootstrap (sequential importance resampling) particle filter. Suited to strongly
nonlinear or non-Gaussian problems; needs more members than the [`EnKF`](@ref).

    ParticleFilter(; N = 500, resample_threshold = 0.5, jitter = 0.0)

# Keywords
- `N`: number of particles.
- `resample_threshold`: resample when the effective sample size drops below this
  fraction of `N`.
- `jitter`: after resampling, perturb particles by this fraction of the posterior
  standard deviation (regularization against sample impoverishment). For an
  [`AgentModel`](@ref), only the estimated parameters are perturbed.

$(TYPEDFIELDS)
"""
struct ParticleFilter <: AbstractEstimator
    N::Int
    resample_threshold::Float64
    jitter::Float64
end

function ParticleFilter(; N::Integer = 500, resample_threshold::Real = 0.5,
        jitter::Real = 0.0)
    0 ≤ resample_threshold ≤ 1 || throw(ArgumentError("resample_threshold must be in [0, 1]"))
    ParticleFilter(Int(N), Float64(resample_threshold), Float64(jitter))
end

initial_belief(::ParticleFilter, X) = ParticleBelief(X)

function update!(f::ParticleFilter, twin::Twin, obs::Observation)
    b = twin.belief::ParticleBelief
    obsidx = observed_mask(obs)
    Y = predicted_observations(twin, b.X)[obsidx, :]
    lik = MvNormal(obs.y[obsidx], Symmetric(twin.obs.R[obsidx, obsidx]))
    for i in axes(Y, 2)
        b.logw[i] += logpdf(lik, Y[:, i])
    end
    all(isinf, b.logw) &&
        throw(ErrorException("all particle weights are zero; the observation is incompatible with every particle"))
    b.logw .-= maximum(b.logw)

    if ess(b) < f.resample_threshold * nmembers(b)
        σ = std(b)
        resample!(b, systematic_resample(twin.rng, weights(b), nmembers(b)))
        fill!(b.logw, 0.0)
        if f.jitter > 0
            # The state rows of a model with latent objects summarize those objects.
            rows = haslatent(twin.model) ? ((nu(twin) + 1):size(b.X, 1)) : axes(b.X, 1)
            b.X[rows, :] .+= (f.jitter .* σ[rows]) .* randn(twin.rng, length(rows), size(b.X, 2))
        end
    end
    return nothing
end
