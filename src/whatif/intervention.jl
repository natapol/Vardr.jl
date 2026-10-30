"""
$(TYPEDEF)

A change applied to the system at a given model time, used for what-if
[`scenario`](@ref)s.
"""
abstract type AbstractIntervention end

"""
$(TYPEDEF)

Set one or more parameters at time `t`, e.g. `SetParameter(10.0, :β => 0.1)`.

$(TYPEDFIELDS)
"""
struct SetParameter <: AbstractIntervention
    t::Float64
    changes::Vector{Pair{Any, Float64}}
end
SetParameter(t::Real, changes::Pair...) = SetParameter(Float64(t),
    Pair{Any, Float64}[k => Float64(v) for (k, v) in changes])

"""
$(TYPEDEF)

Set one or more states at time `t`, e.g. `SetState(5.0, :x => 0.0)`.

$(TYPEDFIELDS)
"""
struct SetState <: AbstractIntervention
    t::Float64
    changes::Vector{Pair{Any, Float64}}
end
SetState(t::Real, changes::Pair...) = SetState(Float64(t),
    Pair{Any, Float64}[k => Float64(v) for (k, v) in changes])

"""
$(TYPEDEF)

Run an arbitrary function at time `t`, for anything that [`SetParameter`](@ref) and
[`SetState`](@ref) cannot express. For an [`ODEModel`](@ref) or
[`NeuralODEModel`](@ref) it is called as `affect!(integrator)`; for an
[`AgentModel`](@ref) as `affect!(sim, p, t)`, with the member's simulation object and
parameter vector.

$(TYPEDFIELDS)
"""
struct Event{F} <: AbstractIntervention
    t::Float64
    affect!::F
end
Event(t::Real, affect!) = Event(Float64(t), affect!)

# Apply an intervention directly to (u, p) when it takes effect at or before the start.
function apply_now!(m::AbstractTwinModel, u, p, iv::SetParameter)
    for (s, v) in iv.changes
        setparam!(m, p, s, v)
    end
end
function apply_now!(m::AbstractTwinModel, u, p, iv::SetState)
    for (s, v) in iv.changes
        u[state_index(m, s)] = v
    end
end
apply_now!(m::AbstractTwinModel, u, p, iv::Event) =
    throw(ArgumentError("an Event must occur after the start of the simulation"))

"Reject interventions that cannot be applied from time `t0`."
function check_interventions(ivs, t0)
    for iv in ivs
        iv isa Event && iv.t <= t0 &&
            throw(ArgumentError("an Event must occur after the start of the simulation (t = $t0)"))
    end
    return nothing
end

# Agent models apply interventions between steps (see `trajectory`).
apply_agent!(m::AgentModel, sim, p, iv::SetParameter, t) = apply_now!(m, nothing, p, iv)
apply_agent!(m::AgentModel, sim, p, iv::Event, t) = iv.affect!(sim, p, t)

affect_fn(m::ODEModel, iv::SetParameter) = integ -> apply_now!(m, integ.u, integ.p, iv)
affect_fn(m::ODEModel, iv::SetState) = integ -> apply_now!(m, integ.u, integ.p, iv)
affect_fn(m::ODEModel, iv::Event) = iv.affect!

"""
Split interventions into those applied immediately (at or before `t0`) and a
callback set (plus `tstops`) for the rest.
"""
function intervention_callbacks(m::ODEModel, ivs, t0)
    check_interventions(ivs, t0)
    now = filter(iv -> iv.t <= t0, ivs)
    later = filter(iv -> iv.t > t0, ivs)
    cbs = map(later) do iv
        tiv = iv.t
        affect! = affect_fn(m, iv)
        DiscreteCallback((u, t, integ) -> t == tiv, affect!; save_positions = (false, false))
    end
    return now, (isempty(cbs) ? nothing : CallbackSet(cbs...)), Float64[iv.t for iv in later]
end
