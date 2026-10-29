"""
$(TYPEDEF)

Strategy for synchronizing a twin's belief with observations. Every estimator
represents the belief by `N` members that are propagated through the model.
"""
abstract type AbstractEstimator end

"""
$(TYPEDEF)

Open-loop twin: members are propagated but observations are ignored.

$(TYPEDFIELDS)
"""
struct NoAssimilation <: AbstractEstimator
    "Ensemble size."
    N::Int
end
NoAssimilation(; N::Integer = 50) = NoAssimilation(N)

initial_belief(::AbstractEstimator, X) = EnsembleBelief(X)

"""
$(TYPEDEF)

A digital twin: a model, a belief about the real system's state, and the machinery
to keep that belief synchronized with observations.

    Twin(model, obs = nothing; estimator = EnKF(), estimate = [], prior = nothing,
         t0 = initial time of the model, process_noise = 0, bounds = nothing,
         rng = Random.default_rng(), name = "twin")

# Keywords
- `estimate`: parameters to estimate jointly with the states (augmented state
  `z = [u; θ]`).
- `prior`: initial belief over `z`. Either a `Distributions` multivariate
  distribution, a matrix whose columns are members, or `nothing` for a Gaussian
  with 10% relative spread around the model's nominal `u0` and parameters.
- `process_noise`: model error, as standard deviation per unit time of each
  component of `z` (scalar or vector). Applied as a random walk during
  [`predict!`](@ref). A small value on estimated parameters keeps their ensemble
  from collapsing.
- `bounds`: `(lower, upper)` vectors over `z`. Members are clamped after every
  prediction and update, for example to keep concentrations non-negative.

`model` is any [`AbstractTwinModel`](@ref). For an [`AgentModel`](@ref), each member
also gets its own simulation, built by the model's `init` from the member's prior
draw. Its state rows then come from the simulation, so `process_noise` and `bounds`
may only be set for the estimated parameters.

$(TYPEDFIELDS)
"""
mutable struct Twin{M <: AbstractTwinModel, O, E <: AbstractEstimator, B <: AbstractBelief, P,
    R <: AbstractRNG}
    name::String
    model::M
    "Observation model, or `nothing` for a twin that is only simulated."
    obs::O
    estimator::E
    belief::B
    "Current model time of the belief."
    t::Float64
    "Parameters estimated online (appended to the state)."
    estimated::Vector{Any}
    "Parameter object shared by all members (estimated entries are overridden per member)."
    p::P
    process_noise::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    history::History
    rng::R
    lock::ReentrantLock
end

function Twin(model::AbstractTwinModel, obs = nothing;
        estimator::AbstractEstimator = EnKF(), estimate = Any[], prior = nothing,
        t0 = initial_time(model), process_noise = 0.0, bounds = nothing,
        rng::AbstractRNG = Random.default_rng(), name::AbstractString = "twin")
    estimate = collect(Any, estimate)
    p = deepcopy(nominal_params(model))
    θ0 = Float64[getparam(model, p, s) for s in estimate]
    z0 = vcat(Float64.(nominal_state(model)), θ0)
    nz = length(z0)
    names = vcat(statenames(model), Symbol[Symbol(string(s)) for s in estimate])

    X = initial_members(rng, prior, z0, estimator.N)
    size(X, 1) == nz ||
        throw(DimensionMismatch("prior has dimension $(size(X, 1)); the augmented state has $nz components"))

    noise = _expand(process_noise, nz, "process_noise")
    lower, upper = bounds === nothing ? (fill(-Inf, nz), fill(Inf, nz)) :
                   (_expand(bounds[1], nz, "lower bound"), _expand(bounds[2], nz, "upper bound"))
    X = clamp.(X, lower, upper)
    haslatent(model) && check_latent_twin(model, estimator, noise, lower, upper)

    twin = Twin(String(name), model, obs, estimator, initial_belief(estimator, X),
        float(t0), estimate, p, noise, lower, upper, History(names), rng,
        ReentrantLock())
    haslatent(model) && init_latent!(twin)
    record!(twin.history, twin.t, :init, twin.belief)
    return twin
end

# The state rows of a model with latent objects are summaries of those objects, so
# they cannot be perturbed or clamped directly.
function check_latent_twin(model, estimator, noise, lower, upper)
    n = nstates(model)
    any(!iszero, noise[1:n]) &&
        throw(ArgumentError("process_noise must be zero for the states of an $(nameof(typeof(model))); its randomness comes from the simulation (noise on estimated parameters is allowed)"))
    all(isinf, lower[1:n]) && all(isinf, upper[1:n]) ||
        throw(ArgumentError("bounds cannot be set on the states of an $(nameof(typeof(model))); bound the estimated parameters only"))
    estimator isa EnKF && !canlift(model) &&
        throw(ArgumentError("the EnKF shifts member states, so an $(nameof(typeof(model))) needs a `lift` function; or use a ParticleFilter"))
    return nothing
end

"Build every member's latent object, and set its state rows from it."
function init_latent!(twin::Twin)
    X = twin.belief.X
    n = nu(twin)
    rngs = member_rngs(twin.model, twin.rng, size(X, 2))
    L = Vector{Any}(undef, size(X, 2))
    tforeach(size(X, 2)) do i
        z = @view X[:, i]
        X[1:n, i], L[i] = init_latent(twin.model, z[1:n], member_params(twin, z), rngs[i])
    end
    twin.belief.latent = L
    return twin
end

"""
After a filter has moved the state rows of a model with latent objects, adjust each
latent object to its new state (and take the state it actually reaches).
"""
function lift_members!(twin::Twin)
    X = twin.belief.X
    n = nu(twin)
    rngs = member_rngs(twin.model, twin.rng, size(X, 2))
    L = twin.belief.latent
    tforeach(size(X, 2)) do i
        z = @view X[:, i]
        X[1:n, i], L[i] = lift(twin.model, L[i], z[1:n], member_params(twin, z), rngs[i])
    end
    return twin
end

function _expand(x, n, what)
    x isa Real && return fill(Float64(x), n)
    length(x) == n || throw(DimensionMismatch("$what has $(length(x)) entries, expected $n"))
    return Float64.(collect(x))
end

function initial_members(rng, prior::Nothing, z0, N)
    σ = 0.1 .* abs.(z0) .+ 1e-8
    return z0 .+ σ .* randn(rng, length(z0), N)
end
initial_members(rng, prior::Distributions.Sampleable, z0, N) = Matrix{Float64}(rand(rng, prior, N))
function initial_members(rng, prior::AbstractMatrix, z0, N)
    size(prior, 2) == N ||
        throw(DimensionMismatch("prior has $(size(prior, 2)) members; the estimator expects $N"))
    return Matrix{Float64}(prior)
end

nu(twin::Twin) = nstates(twin.model)
nθ(twin::Twin) = length(twin.estimated)

"Parameter object for a member with augmented state `z`."
member_params(twin::Twin, z) = withparams(twin.model, twin.p, twin.estimated,
    @view z[(nu(twin) + 1):end])

Base.show(io::IO, twin::Twin) = print(io, "Twin(\"", twin.name, "\", t = ", twin.t, ", ",
    nameof(typeof(twin.estimator)), " with ", nmembers(twin.belief), " members)")

# --- Synchronization -------------------------------------------------------------------

"""
    predict!(twin, t)

Advance every member of the twin's belief from `twin.t` to model time `t`, adding
process noise. Members are integrated in parallel across Julia threads.

If an integration fails, an [`IntegrationError`](@ref) is thrown and the belief of an
ODE twin is left unchanged. (Agent simulations are advanced in place, so after a
failed step their state is undefined.)
"""
function predict!(twin::Twin, t::Real; callback = nothing)
    lock(twin.lock) do
        t < twin.t && throw(ArgumentError("cannot predict backwards from t = $(twin.t) to t = $t"))
        t == twin.t && return twin
        # Propagate into a copy so a failed integration leaves the belief untouched.
        X = copy(twin.belief.X)
        L = copy(twin.belief.latent)
        n = nu(twin)
        t0 = twin.t
        rngs = member_rngs(twin.model, twin.rng, size(X, 2))
        tforeach(size(X, 2)) do i
            z = @view X[:, i]
            X[1:n, i], lat = advance(twin.model, z[1:n], latent_member(twin.belief, i),
                member_params(twin, z), t0, t, rngs[i]; callback)
            isempty(L) || (L[i] = lat)
        end
        twin.belief.X = X
        twin.belief.latent = L
        dt = t - t0
        if any(!iszero, twin.process_noise)
            X .+= (twin.process_noise .* sqrt(dt)) .* randn(twin.rng, size(X)...)
        end
        X .= clamp.(X, twin.lower, twin.upper)
        twin.t = float(t)
        record!(twin.history, twin.t, :predict, twin.belief)
        return twin
    end
end

"""
    assimilate!(twin, obs::Observation)

Update the twin's belief with an observation taken at the twin's current time
(`obs.t == twin.t`). Use [`step!`](@ref) to predict and assimilate in one call.
"""
function assimilate!(twin::Twin, obs::Observation)
    twin.obs === nothing &&
        throw(ArgumentError("this twin has no observation model; pass one to `Twin`"))
    check_observation(twin.obs, obs)
    lock(twin.lock) do
        isapprox(obs.t, twin.t; atol = 1e-10 * max(1, abs(twin.t))) ||
            throw(ArgumentError("observation at t = $(obs.t) but the twin is at t = $(twin.t); call `predict!` first"))
        isempty(observed_mask(obs)) && return twin
        update!(twin.estimator, twin, obs)
        twin.belief.X .= clamp.(twin.belief.X, twin.lower, twin.upper)
        record!(twin.history, twin.t, :update, twin.belief)
        push!(twin.history.observations, obs)
        return twin
    end
end

update!(::NoAssimilation, twin, obs) = nothing

"""
    step!(twin, obs::Observation)

Predict up to `obs.t`, then assimilate `obs`.
"""
function SciMLBase.step!(twin::Twin, obs::Observation)
    lock(twin.lock) do
        predict!(twin, obs.t)
        assimilate!(twin, obs)
    end
end

"Predicted observations `h(u, p, t)` for every member, as columns."
function predicted_observations(twin::Twin, X = twin.belief.X)
    n = nu(twin)
    Y = Matrix{Float64}(undef, noutputs(twin.obs), size(X, 2))
    for i in axes(X, 2)
        z = @view X[:, i]
        Y[:, i] = twin.obs.h(z[1:n], member_params(twin, z), twin.t)
    end
    return Y
end

# --- Queries ---------------------------------------------------------------------------

"Posterior mean of the model states."
state(twin::Twin) = mean(twin.belief)[1:nu(twin)]
"Posterior mean of one state or estimated parameter, by name or index."
state(twin::Twin, s) = mean(twin.belief)[zindex(twin, s)]

"Posterior standard deviation of the model states."
uncertainty(twin::Twin) = std(twin.belief)[1:nu(twin)]
"Posterior standard deviation of one state or estimated parameter."
uncertainty(twin::Twin, s) = std(twin.belief)[zindex(twin, s)]

"""
    paramvalues(twin)

Current parameter values as `name => value` pairs: posterior means for estimated
parameters, fixed values otherwise.
"""
function paramvalues(twin::Twin)
    m = mean(twin.belief)
    syms = collect(Any, user_parameters(twin.model))
    if isempty(syms)
        # Unnamed model: parameters are referred to by index.
        syms = twin.p isa AbstractVector ? collect(eachindex(twin.p)) : Any[]
    end
    est = Dict(string(s) => m[nu(twin) + i] for (i, s) in enumerate(twin.estimated))
    return [(s isa Integer ? s : Symbol(string(s))) =>
                get(() -> Float64(getparam(twin.model, twin.p, s)), est, string(s))
            for s in syms]
end

"The twin's [`History`](@ref)."
history(twin::Twin) = twin.history

"Matrix of belief members (columns) over the augmented state."
members(twin::Twin) = copy(twin.belief.X)

Base.getindex(twin::Twin, s) = state(twin, s)

function zindex(twin::Twin, s)
    s isa Integer && return Int(s)
    i = findfirst(==(Symbol(_strip_t(s))), twin.history.names)
    i === nothing || return i
    return state_index(twin.model, s)
end
