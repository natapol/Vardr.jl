"""
$(TYPEDEF)

A single measurement of the real system at model time `t`.

Entries of `y` that are `NaN` are treated as missing, so a partial sensor reading
can still be assimilated.

    Observation(t, y)

$(TYPEDFIELDS)
"""
struct Observation
    "Model time of the measurement."
    t::Float64
    "Measured values, in the order of the observation model's outputs."
    y::Vector{Float64}
end

Observation(t::Real, y::Real) = Observation(Float64(t), [Float64(y)])
Observation(t::Real, y::AbstractVector) = Observation(Float64(t), Float64.(y))

"""
$(TYPEDEF)

Maps a model state to what the sensors measure: `y = h(u, p, t) + ε`, with
`ε ~ N(0, R)`.

# Constructors

    ObservationModel(model, vars, σ)
    ObservationModel(h, R; names)

The first form observes model states directly. `vars` is a vector of state names
or indices, and `σ` is a scalar or a vector of measurement standard deviations.
The second form takes an arbitrary function `h(u, p, t) -> Vector` and a
measurement covariance matrix `R` (or a vector of standard deviations).

$(TYPEDFIELDS)
"""
struct ObservationModel{H}
    "Measurement function `h(u, p, t)`."
    h::H
    "Measurement noise covariance."
    R::Matrix{Float64}
    "Names of the measured outputs."
    names::Vector{Symbol}
end

function ObservationModel(h, R::AbstractMatrix; names = nothing)
    R = Matrix{Float64}(R)
    size(R, 1) == size(R, 2) || throw(ArgumentError("R must be square"))
    isposdef(Symmetric(R)) || throw(ArgumentError("R must be positive definite"))
    names = names === nothing ? [Symbol(:y, i) for i in 1:size(R, 1)] :
            collect(Symbol, names)
    length(names) == size(R, 1) ||
        throw(ArgumentError("got $(length(names)) names for $(size(R, 1)) outputs"))
    ObservationModel(h, R, names)
end

function ObservationModel(h, σ::AbstractVector{<:Real}; names = nothing)
    ObservationModel(h, Diagonal(float.(σ) .^ 2); names)
end

function ObservationModel(model::AbstractTwinModel, vars::AbstractVector, σ)
    idx = [state_index(model, v) for v in vars]
    σ = σ isa Real ? fill(float(σ), length(idx)) : float.(σ)
    length(σ) == length(idx) ||
        throw(ArgumentError("got $(length(σ)) σ values for $(length(idx)) observed states"))
    names = statenames(model)[idx]
    ObservationModel(StateSelector(idx), Diagonal(σ .^ 2); names)
end

"Observation function that picks state components (a linear `h`)."
struct StateSelector
    idx::Vector{Int}
end
(s::StateSelector)(u, p, t) = u[s.idx]

noutputs(o::ObservationModel) = size(o.R, 1)

function check_observation(o::ObservationModel, obs::Observation)
    length(obs.y) == noutputs(o) ||
        throw(DimensionMismatch("observation has $(length(obs.y)) values; the observation model has $(noutputs(o)) outputs"))
    return nothing
end

"Indices of the non-missing entries of an observation."
observed_mask(obs::Observation) = findall(!isnan, obs.y)
