"""
$(TYPEDEF)

Supertype of the process models a [`Twin`](@ref) can run:

- [`ODEModel`](@ref): an ordinary differential equation (any SciML `ODEProblem`,
  including ModelingToolkit models);
- [`NeuralODEModel`](@ref): an ODE whose right-hand side is, or contains, a neural
  network (a neural ODE or universal differential equation);
- [`AgentModel`](@ref): a stochastic agent-based simulation, tracked through summary
  quantities.

Every model exposes a *state vector* `u` with named components and a *parameter
object* `p`. Filters, forecasts and calibration only see `u` and `p`. A model can
also give each member a *latent* object that holds everything else, such as the
individual agents of an agent-based model.

# Implementing a new model type

A subtype `M <: AbstractTwinModel` defines:

| Method | Returns |
|:--|:--|
| `nstates(m)`, `statenames(m)` | size and names of `u` |
| `user_parameters(m)` | parameter names (empty if parameters are referred to by index) |
| `nominal_state(m)`, `nominal_params(m)`, `initial_time(m)` | nominal `u0`, `p` and `t0` |
| `getparam(m, p, s)`, `setparam!(m, p, s, v)` | read and write one parameter of `p` |
| `advance(m, u, latent, p, t0, t1, rng; callback)` | `(u, latent)` at `t1` |
| `trajectory(m, u, latent, p, t0, ts, rng; interventions, strict)` | the states at times `ts` |

Models with a latent object also define `haslatent(m) = true`,
`init_latent(m, u, p, rng)` returning a consistent `(u, latent)` pair and, to be
usable with the [`EnKF`](@ref), `canlift(m) = true` and
`lift(m, latent, u, p, rng)`, which adjusts `latent` to match an updated `u`.
"""
abstract type AbstractTwinModel end

"""
    IntegrationError(msg)

A member's simulation failed, for example because the ODE solver stopped early.
"""
struct IntegrationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IntegrationError) = print(io, "IntegrationError: ", e.msg)

"Number of state variables of the model."
function nstates end

"State names of the model (`Symbol`s, or `u1, u2, …` when the model is unnamed)."
function statenames end

"Parameter names of the model (empty when the model is unnamed)."
paramnames(m::AbstractTwinModel) = [Symbol(string(s)) for s in user_parameters(m)]

# MTK names states like `x(t)`; display them as `x`.
_strip_t(s) = replace(string(s), r"\(t\)$" => "")

"Resolve a state reference (`Int` or name) to its index in `u`."
function state_index(m::AbstractTwinModel, s)
    s isa Integer && return Int(s)
    i = findfirst(==(Symbol(_strip_t(s))), statenames(m))
    i === nothing && throw(ArgumentError("unknown state `$s`"))
    return i
end

"Return a copy of the parameter object `p` with parameters `syms` set to `vals`."
function withparams(m::AbstractTwinModel, p, syms, vals)
    isempty(syms) && return p
    # Promote a plain parameter vector so that `vals` may hold dual numbers (gradients
    # for calibration).
    q = p isa AbstractVector{<:Number} ?
        copyto!(similar(p, promote_type(eltype(p), eltype(vals))), p) : deepcopy(p)
    for (s, v) in zip(syms, vals)
        setparam!(m, q, s, v)
    end
    return q
end

"Whether members of this model carry a latent object besides the state vector `u`."
haslatent(::AbstractTwinModel) = false

"Whether `lift` can adjust a member's latent object to a new state vector."
canlift(::AbstractTwinModel) = false

"""
Times from `t0` to `tend` for a forecast: `saveat` if given, else 101 points.
"""
output_times(saveat, t0, tend) = saveat === nothing ?
    collect(range(t0, tend; length = 101)) : collect(Float64, saveat)

"""
Run `f(i)` for `i in 1:n` across threads. An exception from any iteration is rethrown
as-is instead of wrapped in a `CompositeException`.
"""
function tforeach(f, n::Integer)
    try
        Threads.@threads for i in 1:n
            f(i)
        end
    catch e
        e isa CompositeException && !isempty(e.exceptions) || rethrow()
        inner = first(e.exceptions)
        throw(inner isa TaskFailedException ? inner.task.exception : inner)
    end
    return nothing
end

"""
One independent random number generator per member, seeded from `rng`, so that
stochastic members give the same results however they are spread over threads.
`nothing` for deterministic models (and then `rng` is not advanced).
"""
member_rngs(m::AbstractTwinModel, rng::AbstractRNG, n::Integer) =
    haslatent(m) ? [Xoshiro(s) for s in rand(rng, UInt64, n)] : fill(nothing, n)
