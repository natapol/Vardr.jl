"""
$(TYPEDEF)

The mechanistic core of a twin: an `ODEProblem` plus the solver used to advance it.

Any `ODEProblem` works, including one built from a ModelingToolkit system
(`ODEProblem(sys, u0map, tspan, pmap)`), in which case states and parameters can be
referred to by their symbolic names.

# Constructors

    TwinModel(prob::ODEProblem; alg = Tsit5(), solve_kwargs...)
    TwinModel(f, u0, p; states = nothing, params = nothing, t0 = 0.0, alg = Tsit5(), solve_kwargs...)

The second form wraps an out-of-place right-hand side `f(u, p, t)` (or in-place
`f(du, u, p, t)`). Pass `states`/`params` as vectors of `Symbol`s to enable
name-based indexing without ModelingToolkit.

$(TYPEDFIELDS)
"""
struct TwinModel{P <: ODEProblem, A, K}
    "Template problem; its `u0`, `p` and `tspan[1]` are the nominal initial conditions."
    prob::P
    "ODE solver algorithm."
    alg::A
    "Extra keyword arguments forwarded to `solve`."
    solve_kwargs::K
end

function TwinModel(prob::ODEProblem; alg = Tsit5(), solve_kwargs...)
    # Members carry their own (already consistent) states. Without `NoInit`, a
    # ModelingToolkit problem would re-run its initialization system and reset `u0`
    # to the problem's original initial conditions.
    kw = merge((initializealg = SciMLBase.NoInit(),), NamedTuple(solve_kwargs))
    TwinModel(prob, alg, kw)
end

function TwinModel(f, u0, p; states = nothing, params = nothing, t0 = 0.0,
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
    TwinModel(ODEProblem(fun, u0, tspan, p); alg, solve_kwargs...)
end

Base.show(io::IO, m::TwinModel) = print(io, "TwinModel(", nstates(m), " states ",
    statenames(m), ", ", length(m.prob.p isa AbstractVector ? m.prob.p : paramnames(m)),
    " parameters, ", nameof(typeof(m.alg)), ")")

"Number of state variables of the model."
nstates(m::TwinModel) = length(m.prob.u0)

"State names of the model (`Symbol`s, or `u1, u2, …` when the model is unnamed)."
function statenames(m::TwinModel)
    syms = variable_symbols(m.prob)
    isempty(syms) ? [Symbol(:u, i) for i in 1:nstates(m)] :
    [Symbol(_strip_t(s)) for s in syms]
end

"Parameter names of the model (empty when the model is unnamed)."
paramnames(m::TwinModel) = [Symbol(string(s)) for s in user_parameters(m)]

# Parameter symbols, minus the `Initial(x)` parameters ModelingToolkit adds internally.
user_parameters(m::TwinModel) = filter(s -> !startswith(string(s), "Initial("),
    collect(parameter_symbols(m.prob)))

# MTK names states like `x(t)`; display them as `x`.
_strip_t(s) = replace(string(s), r"\(t\)$" => "")

"Resolve a state reference (`Int`, `Symbol`, or symbolic variable) to its index in `u`."
function state_index(m::TwinModel, s)
    s isa Integer && return Int(s)
    i = variable_index(m.prob, s)
    if i === nothing && s isa Symbol
        i = findfirst(==(s), statenames(m))
    end
    i === nothing && throw(ArgumentError("unknown state `$s`"))
    return i
end

"Read parameter `s` from a parameter object (or anything with `parameter_values`)."
getparam(m::TwinModel, p, s) = getp(m.prob, s)(p)

"Return a copy of the parameter object `p` with parameters `syms` set to `vals`."
function withparams(m::TwinModel, p, syms, vals)
    isempty(syms) && return p
    q = deepcopy(p)
    for (s, v) in zip(syms, vals)
        setp(m.prob, s)(q, v)
    end
    return q
end

"""
The model's problem with a member's state, parameters and time span.

`build_initializeprob = false` stops ModelingToolkit from recomputing `u0` from its
`Initial(...)` parameters when an explicit parameter object is passed.
"""
member_problem(m::TwinModel, u, p, tspan) = remake(m.prob; u0 = u, p, tspan,
    build_initializeprob = false)

"""
Solve a member problem with the model's solver and keyword arguments.

`callback` and `tstops` are combined with any the model was built with (for example a
dosing schedule), never substituted for them.
"""
function solve_member(m::TwinModel, prob; callback = nothing, tstops = Float64[],
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
function propagate(m::TwinModel, u, p, t0, t1; callback = nothing)
    t1 == t0 && return copy(u)
    prob = member_problem(m, copy(u), p, (t0, t1))
    sol = solve_member(m, prob; callback, save_everystep = false, save_start = false,
        save_end = true)
    SciMLBase.successful_retcode(sol) ||
        throw(ErrorException("model integration failed with retcode $(sol.retcode)"))
    return collect(sol.u[end])
end

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
