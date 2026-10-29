"""
$(TYPEDEF)

A stochastic agent-based (individual-based) model, stepped in fixed time steps.

    AgentModel(init, step!, summarize, u0, p; states = nothing, params = nothing,
        dt = 1.0, t0 = 0.0, lift = nothing)

Each member of the twin's belief owns a complete simulation, `sim`: the agents and
anything else the model needs. The twin tracks the simulation through a short
*summary state* `u = summarize(sim)`, such as the number of agents in each
compartment. Observation models, forecasts, the history and the filters work on `u`;
the simulation is the member's *latent* state.

# Arguments
- `init(u, p, rng) -> sim`: build one member's simulation for parameters `p`.
  `u` is a summary state drawn from the twin's prior; use it to set the initial
  condition (for example round it to agent counts), or ignore it.
- `step!(sim, p, t, rng)`: advance `sim` in place by one step of length `dt`,
  starting at time `t`. Draw all randomness from `rng`.
- `summarize(sim) -> Vector`: the summary state; its length must match `u0`.
- `u0`, `p`: nominal summary state and parameter vector.

# Keywords
- `states`, `params`: names of the summary components and parameters.
- `dt`: time step. Observation and forecast times must lie on the step grid.
- `lift(sim, u, p, rng) -> sim`: optional. Change a simulation so that its summary
  becomes (approximately) `u`, for example by moving agents between compartments.
  Only the [`EnKF`](@ref) needs it, because it shifts members' states directly. The
  [`ParticleFilter`](@ref) only reweights and copies whole simulations, so it works
  with any agent model.

Each member gets its own random number generator for every prediction, so results
are reproducible from the twin's `rng` and independent of the number of threads.
Members that the particle filter duplicates are deep copies that then evolve
independently.

$(TYPEDFIELDS)
"""
struct AgentModel{I, S, F, L} <: AbstractTwinModel
    init::I
    step!::S
    summarize::F
    "Optional `lift(sim, u, p, rng)`, or `nothing`."
    lift::L
    u0::Vector{Float64}
    p::Vector{Float64}
    states::Vector{Symbol}
    params::Vector{Symbol}
    dt::Float64
    t0::Float64
end

function AgentModel(init, step!, summarize, u0, p; states = nothing, params = nothing,
        dt::Real = 1.0, t0::Real = 0.0, lift = nothing)
    u0 = collect(Float64, u0)
    p = collect(Float64, p)
    states = states === nothing ? [Symbol(:u, i) for i in eachindex(u0)] :
             collect(Symbol, states)
    params = params === nothing ? [Symbol(:p, i) for i in eachindex(p)] :
             collect(Symbol, params)
    length(states) == length(u0) ||
        throw(ArgumentError("`states` has $(length(states)) names but u0 has $(length(u0)) entries"))
    length(params) == length(p) ||
        throw(ArgumentError("`params` has $(length(params)) names but p has $(length(p)) entries"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    AgentModel(init, step!, summarize, lift, u0, p, states, params, Float64(dt),
        Float64(t0))
end

Base.show(io::IO, m::AgentModel) = print(io, "AgentModel(", nstates(m), " summary states ",
    m.states, ", ", length(m.p), " parameters, dt = ", m.dt, ")")

nstates(m::AgentModel) = length(m.u0)
statenames(m::AgentModel) = m.states
user_parameters(m::AgentModel) = m.params
nominal_state(m::AgentModel) = copy(m.u0)
nominal_params(m::AgentModel) = m.p
initial_time(m::AgentModel) = m.t0
haslatent(::AgentModel) = true
canlift(m::AgentModel) = m.lift !== nothing

function param_index(m::AgentModel, s)
    s isa Integer && return Int(s)
    i = findfirst(==(Symbol(string(s))), m.params)
    i === nothing && throw(ArgumentError("unknown parameter `$s`"))
    return i
end
getparam(m::AgentModel, p, s) = p[param_index(m, s)]
setparam!(m::AgentModel, p, s, v) = (p[param_index(m, s)] = v; p)

function summary_state(m::AgentModel, sim)
    u = collect(Float64, m.summarize(sim))
    length(u) == nstates(m) ||
        throw(DimensionMismatch("summarize returned $(length(u)) values; the model has $(nstates(m)) summary states"))
    return u
end

init_latent(m::AgentModel, u, p, rng) = (sim = m.init(u, p, rng); (summary_state(m, sim), sim))

function lift(m::AgentModel, sim, u, p, rng)
    canlift(m) || throw(ArgumentError("this AgentModel has no `lift` function"))
    sim = m.lift(sim, u, p, rng)
    return summary_state(m, sim), sim
end

"Number of model steps from `t0` to `t1`; the interval must be a multiple of `dt`."
function nsteps(m::AgentModel, t0, t1)
    r = (t1 - t0) / m.dt
    k = round(Int, r)
    isapprox(r, k; atol = 1e-8 * max(1, abs(r))) && k >= 0 ||
        throw(ArgumentError("the agent model steps by dt = $(m.dt); cannot go from t = $t0 to t = $t1"))
    return k
end

function advance(m::AgentModel, u, sim, p, t0, t1, rng; callback = nothing)
    callback === nothing ||
        throw(ArgumentError("agent models do not take SciML callbacks; use interventions"))
    for k in 1:nsteps(m, t0, t1)
        m.step!(sim, p, t0 + (k - 1) * m.dt, rng)
    end
    return summary_state(m, sim), sim
end

function trajectory(m::AgentModel, u, sim, p, t0, ts, rng;
        interventions = AbstractIntervention[], strict::Bool = true)
    p = copy(p)
    sim === nothing && ((_, sim) = init_latent(m, u, p, rng))
    check_interventions(interventions, t0)
    any(iv -> iv isa SetState, interventions) &&
        throw(ArgumentError("SetState is not available for agent models; use an Event that changes the agents"))
    pending = sort(collect(interventions); by = iv -> iv.t)
    k = 0                        # steps taken; the simulation is at t0 + k * dt
    out = Vector{Vector{Float64}}(undef, length(ts))
    for (j, tj) in enumerate(ts)
        for _ in 1:nsteps(m, t0 + k * m.dt, tj)
            t = t0 + k * m.dt
            # Interventions take effect at the first step at or after their time.
            while !isempty(pending) && first(pending).t <= t + 1e-9 * m.dt
                apply_agent!(m, sim, p, popfirst!(pending), t)
            end
            m.step!(sim, p, t, rng)
            k += 1
        end
        out[j] = summary_state(m, sim)
    end
    return out
end
