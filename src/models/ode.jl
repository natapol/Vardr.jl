"""
$(TYPEDEF)

A process model given by an ordinary differential equation: an `ODEProblem` plus the
solver used to advance it.

Any `ODEProblem` works, including one built from a ModelingToolkit system
(`ODEProblem(sys, u0map, tspan, pmap)`), in which case states and parameters can be
referred to by their symbolic names.

# Constructors

    ODEModel(prob::ODEProblem; alg = Tsit5(), solve_kwargs...)
    ODEModel(f, u0, p; states = nothing, params = nothing, t0 = 0.0, alg = Tsit5(), solve_kwargs...)

The second form wraps an out-of-place right-hand side `f(u, p, t)` (or in-place
`f(du, u, p, t)`). Pass `states`/`params` as vectors of `Symbol`s to enable
name-based indexing without ModelingToolkit.

$(TYPEDFIELDS)
"""
struct ODEModel{P <: ODEProblem, A, K} <: AbstractTwinModel
    "Template problem; its `u0`, `p` and `tspan[1]` are the nominal initial conditions."
    prob::P
    "ODE solver algorithm."
    alg::A
    "Extra keyword arguments forwarded to `solve`."
    solve_kwargs::K
end

function ODEModel(prob::ODEProblem; alg = Tsit5(), solve_kwargs...)
    # Members carry their own (already consistent) states. Without `NoInit`, a
    # ModelingToolkit problem would re-run its initialization system and reset `u0`
    # to the problem's original initial conditions.
    kw = merge((initializealg = SciMLBase.NoInit(),), NamedTuple(solve_kwargs))
    ODEModel(prob, alg, kw)
end

function ODEModel(f, u0, p; states = nothing, params = nothing, t0 = 0.0,
        alg = Tsit5(), solve_kwargs...)
    u0 = collect(float.(u0))
    tspan = (float(t0), float(t0) + 1)
    fun = if states === nothing && params === nothing
        ODEFunction(f)
    else
        states = states === nothing ? Symbol[Symbol(:u, i) for i in eachindex(u0)] :
                 collect(Symbol, states)
        params = params === nothing ? Symbol[Symbol(:p, i) for i in eachindex(p)] :
                 collect(Symbol, params)
        length(states) == length(u0) ||
            throw(ArgumentError("`states` has $(length(states)) names but u0 has $(length(u0)) entries"))
        ODEFunction(f; sys = SymbolCache(states, params, :t))
    end
    ODEModel(ODEProblem(fun, u0, tspan, p); alg, solve_kwargs...)
end

Base.show(io::IO, m::ODEModel) = print(io, "ODEModel(", nstates(m), " states ",
    statenames(m), ", ", length(m.prob.p isa AbstractVector ? m.prob.p : paramnames(m)),
    " parameters, ", nameof(typeof(m.alg)), ")")

nstates(m::ODEModel) = length(m.prob.u0)

function statenames(m::ODEModel)
    syms = variable_symbols(m.prob)
    isempty(syms) ? [Symbol(:u, i) for i in 1:nstates(m)] :
    [Symbol(_strip_t(s)) for s in syms]
end

# Parameter symbols, minus the `Initial(x)` parameters ModelingToolkit adds internally.
user_parameters(m::ODEModel) = filter(s -> !startswith(string(s), "Initial("),
    collect(parameter_symbols(m.prob)))

nominal_state(m::ODEModel) = Float64.(m.prob.u0)
nominal_params(m::ODEModel) = m.prob.p
initial_time(m::ODEModel) = float(m.prob.tspan[1])

"Resolve a state reference (`Int`, `Symbol`, or symbolic variable) to its index in `u`."
function state_index(m::ODEModel, s)
    s isa Integer && return Int(s)
    i = variable_index(m.prob, s)
    if i === nothing && s isa Symbol
        i = findfirst(==(s), statenames(m))
    end
    i === nothing && throw(ArgumentError("unknown state `$s`"))
    return i
end

"Read parameter `s` from a parameter object (or anything with `parameter_values`)."
getparam(m::ODEModel, p, s) = getp(m.prob, s)(p)

"Set parameter `s` of the parameter object `p` to `v`, in place."
setparam!(m::ODEModel, p, s, v) = setp(m.prob, s)(p, v)

"""
The model's problem with a member's state, parameters and time span.

`build_initializeprob = false` stops ModelingToolkit from recomputing `u0` from its
`Initial(...)` parameters when an explicit parameter object is passed.
"""
member_problem(m::ODEModel, u, p, tspan) = remake(m.prob; u0 = u, p, tspan,
    build_initializeprob = false)

"""
Solve a member problem with the model's solver and keyword arguments.

`callback` and `tstops` are combined with any the model was built with (for example a
dosing schedule), never substituted for them.
"""
function solve_member(m::ODEModel, prob; callback = nothing, tstops = Float64[],
        kwargs...)
    kw = m.solve_kwargs
    cbs = filter(!isnothing, Any[get(kw, :callback, nothing), callback])
    cb = isempty(cbs) ? nothing : length(cbs) == 1 ? only(cbs) : CallbackSet(cbs...)
    t0, t1 = prob.tspan
    ts = Float64[t for t in vcat(collect(get(kw, :tstops, Float64[])), collect(tstops))
                 if t0 < t <= t1]
    solve(prob, m.alg; kw..., kwargs..., callback = cb, tstops = sort!(unique!(ts)))
end

"""
The saved state at each time in `ts`. Callbacks may save extra points (for example
just before and after a dose), so look states up by time, taking the last value saved
at each time (the post-event state).
"""
function states_at(sol, ts)
    return map(ts) do t
        j = findlast(==(t), sol.t)
        j === nothing && throw(ErrorException("the solution has no saved state at t = $t"))
        sol.u[j]
    end
end

"""
Integrate `u` from `t0` to `t1` with parameters `p` and return the final state.
"""
function propagate(m::ODEModel, u, p, t0, t1; callback = nothing)
    t1 == t0 && return copy(u)
    prob = member_problem(m, copy(u), p, (t0, t1))
    sol = solve_member(m, prob; callback, save_everystep = false, save_start = false,
        save_end = true)
    SciMLBase.successful_retcode(sol) ||
        throw(IntegrationError("model integration failed with retcode $(sol.retcode)"))
    return collect(sol.u[end])
end

advance(m::ODEModel, u, latent, p, t0, t1, rng; callback = nothing) =
    (propagate(m, u, p, t0, t1; callback), nothing)

"""
States at times `ts` (sorted, from `t0` on) of a simulation started from `(u, p)` at
`t0`, with what-if `interventions`. With `strict = false`, a failed integration returns
`nothing` instead of throwing.
"""
function trajectory(m::ODEModel, u, latent, p, t0, ts, rng;
        interventions = AbstractIntervention[], strict::Bool = true)
    u = copy(u)
    p = deepcopy(p)
    now, callback, tstops = intervention_callbacks(m, interventions, t0)
    for iv in now
        apply_now!(m, u, p, iv)
    end
    t1 = last(ts) > t0 ? float(last(ts)) : t0 + 1
    sol = solve_member(m, member_problem(m, u, p, (t0, t1)); saveat = ts, callback, tstops)
    if !SciMLBase.successful_retcode(sol)
        strict || return nothing
        throw(IntegrationError("model integration failed with retcode $(sol.retcode)"))
    end
    return states_at(sol, ts)
end
