"""
$(TYPEDEF)

A probabilistic description of the twin's (augmented) state `z = [u; θ]`, where
`u` are the model states and `θ` are the parameters being estimated online.

Members of an [`AgentModel`](@ref) also carry a latent simulation object each, in
`belief.latent` (empty for other models).
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
    "Latent object of each member (empty when the model has none)."
    latent::Vector{Any}
end
EnsembleBelief(X::Matrix{Float64}) = EnsembleBelief(X, Any[])

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
    "Latent object of each particle (empty when the model has none)."
    latent::Vector{Any}
end

ParticleBelief(X::Matrix{Float64}) = ParticleBelief(X, zeros(size(X, 2)), Any[])

"Latent object of member `i`, or `nothing` when the model has none."
latent_member(b::AbstractBelief, i) = isempty(b.latent) ? nothing : b.latent[i]

"""
Keep members `idx` (with repeats), as in resampling. A latent object chosen more than
once is deep-copied, so that the copies evolve independently.
"""
function resample!(b::AbstractBelief, idx)
    b.X = b.X[:, idx]
    if !isempty(b.latent)
        used = falses(length(b.latent))
        b.latent = map(idx) do j
            used[j] && return deepcopy(b.latent[j])
            used[j] = true
            b.latent[j]
        end
    end
    return b
end

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
Indices of `n` equally weighted members drawn from the belief (all of them, in order,
when `n` equals the number of members of an `EnsembleBelief`).
"""
function draw(rng::AbstractRNG, b::EnsembleBelief, n::Integer)
    n == nmembers(b) && return collect(1:n)
    return rand(rng, 1:nmembers(b), n)
end
draw(rng::AbstractRNG, b::ParticleBelief, n::Integer) = systematic_resample(rng, weights(b), n)

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
