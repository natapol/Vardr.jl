"""
$(TYPEDEF)

Stochastic ensemble Kalman filter (perturbed observations).

    EnKF(; N = 50, inflation = 1.0, localization = nothing)

# Keywords
- `N`: ensemble size.
- `inflation`: multiplicative inflation of the ensemble anomalies before each update
  (values slightly above 1, such as 1.02, counteract ensemble collapse).
- `localization`: Schur-product covariance localization for the model states, given
  as a tuple `(Lxy, Lyy)`. `Lxy` (`nu × ny`) tapers the state–observation covariance
  and `Lyy` (`ny × ny`) the observation–observation covariance, where `nu` is the
  number of model states and `ny` the number of observed outputs. Both must come
  from the same correlation function, so that the state gain is the exact Kalman gain
  of the localized covariance. Tapering only `Lxy` can inflate the spread at the
  sensors and make the filter diverge. If the observation model observes states
  directly (`ObservationModel(model, vars, σ)`), passing `Lxy` alone is enough: `Lyy`
  is then the rows of `Lxy` at the observed states.

  Estimated parameters are global quantities, so they are always updated with the
  unlocalized gain.

$(TYPEDFIELDS)
"""
struct EnKF{L} <: AbstractEstimator
    N::Int
    inflation::Float64
    localization::L
end

function EnKF(; N::Integer = 50, inflation::Real = 1.0, localization = nothing)
    N ≥ 2 || throw(ArgumentError("EnKF needs at least 2 members"))
    inflation ≥ 1 || throw(ArgumentError("inflation must be ≥ 1"))
    EnKF(Int(N), Float64(inflation), localization)
end

# The (Lxy, Lyy) localization pair, completing Lyy for direct state observations.
localization_pair(L::Tuple{AbstractMatrix, AbstractMatrix}, obsmodel) = L
function localization_pair(Lxy::AbstractMatrix, obsmodel)
    obsmodel.h isa StateSelector ||
        throw(ArgumentError("localization needs (Lxy, Lyy) when the observation model is not a selection of states"))
    return (Lxy, Lxy[obsmodel.h.idx, :])
end

function update!(f::EnKF, twin::Twin, obs::Observation)
    X = twin.belief.X
    N = size(X, 2)
    if f.inflation != 1
        x̄ = mean(X; dims = 2)
        X .= x̄ .+ f.inflation .* (X .- x̄)
    end
    obsidx = observed_mask(obs)
    Y = predicted_observations(twin, X)[obsidx, :]
    y = obs.y[obsidx]
    R = twin.obs.R[obsidx, obsidx]

    A = X .- mean(X; dims = 2)
    B = Y .- mean(Y; dims = 2)
    Cxy = A * B' ./ (N - 1)
    Cyy = B * B' ./ (N - 1)

    # Perturbed observations: each member assimilates y + ε_i, ε_i ~ N(0, R)
    E = cholesky(Symmetric(R)).L * randn(twin.rng, length(y), N)
    D = (y .+ E) .- Y
    innovation = cholesky(Symmetric(Cyy + R)) \ D
    if f.localization === nothing
        X .+= Cxy * innovation
    else
        n = nu(twin)
        Lxy, Lyy = localization_pair(f.localization, twin.obs)
        size(Lxy, 1) >= n ||
            throw(DimensionMismatch("Lxy has $(size(Lxy, 1)) rows; the model has $n states"))
        Cuy = Cxy[1:n, :] .* Lxy[1:n, obsidx]
        X[1:n, :] .+= Cuy * (cholesky(Symmetric(Cyy .* Lyy[obsidx, obsidx] + R)) \ D)
        X[(n + 1):end, :] .+= Cxy[(n + 1):end, :] * innovation
    end
    return nothing
end
