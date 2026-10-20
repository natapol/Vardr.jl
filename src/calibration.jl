"""
    calibrate!(twin, data, params; alg, lb = nothing, ub = nothing, adtype = SciMLBase.NoAD(), u0 = state(twin), solve_kwargs...)

Fit model parameters `params` (names or indices) to a batch of observations by
maximizing the Gaussian likelihood under the twin's observation model, simulating
from `u0` at `twin.t`. All observations must be at or after `twin.t`.

`alg` is any Optimization.jl optimizer, e.g. `NelderMead()` after
`using OptimizationOptimJL`. Gradient-based optimizers need an `adtype` such as
`AutoForwardDiff()`.

On success the fitted values replace the twin's fixed parameters. If a parameter is
also estimated online, its member values are shifted so that their mean equals the
fitted value. Returns the Optimization.jl solution.
"""
function calibrate!(twin::Twin, data::AbstractVector{Observation}, params;
        alg, lb = nothing, ub = nothing, adtype = SciMLBase.NoAD(), u0 = state(twin),
        solve_kwargs...)
    twin.obs === nothing && throw(ArgumentError("calibration needs an observation model"))
    isempty(data) && throw(ArgumentError("no observations to calibrate against"))
    params = collect(Any, params)
    data = sort(data; by = o -> o.t)
    foreach(o -> check_observation(twin.obs, o), data)
    first(data).t ≥ twin.t ||
        throw(ArgumentError("observations must be at or after the twin's time t = $(twin.t)"))

    m = twin.model
    ts = unique([o.t for o in data])
    Rinv = [inv(Symmetric(twin.obs.R[mask, mask])) for mask in observed_mask.(data)]
    base_p = member_params(twin, mean(twin.belief))
    tend = last(ts)
    tspan = (twin.t, tend == twin.t ? tend + 1 : tend)

    function loss(θ, _)
        p = withparams(m, base_p, params, θ)
        prob = member_problem(m, collect(u0), p, tspan)
        sol = solve_member(m, prob; saveat = ts)
        SciMLBase.successful_retcode(sol) || return Inf
        us = states_at(sol, ts)
        nll = zero(eltype(θ))
        for (o, Ri) in zip(data, Rinv)
            j = searchsortedfirst(ts, o.t)
            mask = observed_mask(o)
            r = o.y[mask] .- twin.obs.h(us[j], p, o.t)[mask]
            nll += dot(r, Ri, r) / 2
        end
        return nll
    end

    θ0 = Float64[getparam(m, base_p, s) for s in params]
    optf = Optimization.OptimizationFunction(loss, adtype)
    optprob = Optimization.OptimizationProblem(optf, θ0; lb, ub)
    sol = Optimization.solve(optprob, alg; solve_kwargs...)

    lock(twin.lock) do
        θ̂ = collect(sol.u)
        twin.p = withparams(m, twin.p, params, θ̂)
        z̄ = mean(twin.belief)
        for (s, v) in zip(params, θ̂)
            k = findfirst(e -> isequal(e, s), twin.estimated)
            k === nothing && continue
            row = @view twin.belief.X[nu(twin) + k, :]
            row .+= v - z̄[nu(twin) + k]
        end
    end
    return sol
end
