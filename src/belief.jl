"""
$(TYPEDEF)

A probabilistic description of the twin's (augmented) state `z = [u; θ]`, where
`u` are the model states and `θ` are the parameters being estimated online.
"""
abstract type AbstractBelief end

"""
$(TYPEDEF)

Equally weighted sample of the augmented state; column `i` of `X` is member `i`.

$(TYPEDFIELDS)
"""
mutable struct EnsembleBelief <: AbstractBelief
    "Members as columns."
    X::Matrix{Float64}
end

"""
$(TYPEDEF)

Weighted sample of the augmented state.

$(TYPEDFIELDS)
"""
mutable struct ParticleBelief <: AbstractBelief
    "Particles as columns."
    X::Matrix{Float64}
    "Log importance weights (unnormalized)."
    logw::Vector{Float64}
end

ParticleBelief(X::Matrix{Float64}) = ParticleBelief(X, zeros(size(X, 2)))

nmembers(b::AbstractBelief) = size(b.X, 2)
members(b::AbstractBelief) = b.X

weights(::EnsembleBelief) = nothing
function weights(b::ParticleBelief)
    w = exp.(b.logw .- maximum(b.logw))
    return w ./ sum(w)
end

Statistics.mean(b::EnsembleBelief) = vec(mean(b.X; dims = 2))
Statistics.mean(b::ParticleBelief) = b.X * weights(b)

Statistics.cov(b::EnsembleBelief) = nmembers(b) > 1 ? cov(b.X; dims = 2) :
                                    zeros(size(b.X, 1), size(b.X, 1))
function Statistics.cov(b::ParticleBelief)
    w = weights(b)
    A = b.X .- mean(b)
    return (A .* w') * A' ./ (1 - sum(abs2, w))
end

Statistics.std(b::AbstractBelief) = sqrt.(max.(diag(cov(b)), 0.0))

"Effective sample size of a weighted belief."
ess(b::ParticleBelief) = 1 / sum(abs2, weights(b))
ess(b::EnsembleBelief) = float(nmembers(b))

"""
Return `n` equally weighted columns drawn from the belief (all of them, unweighted,
when `n` equals the number of members of an `EnsembleBelief`).
"""
function draw(rng::AbstractRNG, b::EnsembleBelief, n::Integer)
    n == nmembers(b) && return copy(b.X)
    return b.X[:, rand(rng, 1:nmembers(b), n)]
end
draw(rng::AbstractRNG, b::ParticleBelief, n::Integer) = b.X[:, systematic_resample(rng, weights(b), n)]

"Systematic resampling: `n` indices drawn proportionally to normalized weights `w`."
function systematic_resample(rng::AbstractRNG, w::AbstractVector, n::Integer = length(w))
    idx = Vector{Int}(undef, n)
    c = cumsum(w)
    c[end] = 1.0
    u = rand(rng) / n
    j = 1
    for i in 1:n
        while u > c[j]
            j += 1
        end
        idx[i] = j
        u += 1 / n
    end
    return idx
end
