"""
$(TYPEDEF)

Probabilistic trajectory of the twin's states produced by [`forecast`](@ref) or
[`scenario`](@ref).

- `mean(fc)`, `quantile(fc, q)`, `std(fc)`: `nstates × length(fc.t)` summaries
- `fc[:x]`: `length(fc.t) × nsamples` matrix of sample trajectories of state `x`

$(TYPEDFIELDS)
"""
struct TwinForecast
    name::String
    "Output times."
    t::Vector{Float64}
    "State names."
    names::Vector{Symbol}
    "Sample trajectories, `nstates × length(t) × nsamples`."
    samples::Array{Float64, 3}
end

Base.show(io::IO, fc::TwinForecast) = print(io, "TwinForecast(\"", fc.name, "\", t ∈ [",
    first(fc.t), ", ", last(fc.t), "], ", size(fc.samples, 3), " samples)")

Statistics.mean(fc::TwinForecast) = dropdims(mean(fc.samples; dims = 3); dims = 3)
Statistics.std(fc::TwinForecast) = dropdims(std(fc.samples; dims = 3); dims = 3)
function Statistics.quantile(fc::TwinForecast, q::Real)
    [quantile(view(fc.samples, i, j, :), q) for i in axes(fc.samples, 1),
     j in axes(fc.samples, 2)]
end

function Base.getindex(fc::TwinForecast, s)
    i = s isa Integer ? Int(s) : findfirst(==(Symbol(_strip_t(s))), fc.names)
    i === nothing && throw(ArgumentError("unknown state `$s`"))
    return fc.samples[i, :, :]
end

"""
    forecast(twin, tend; saveat = range(twin.t, tend; length = 101), nsamples, interventions = [], name = "forecast", rng)

Propagate samples from the twin's current belief to `tend` without changing the
twin. Returns a [`TwinForecast`](@ref). `nsamples` defaults to the number of
members. Each sample's estimated parameters are those of the member it was drawn
from, so parameter uncertainty is carried into the forecast. Process noise is not
added.
"""
function forecast(twin::Twin, tend::Real; saveat = nothing, nsamples::Integer = nmembers(twin.belief),
        interventions = AbstractIntervention[], name::AbstractString = "forecast",
        rng::AbstractRNG = twin.rng)
    t0 = twin.t
    tend > t0 || throw(ArgumentError("tend = $tend must be after the twin's time t = $t0"))
    ts = saveat === nothing ? collect(range(t0, tend; length = 101)) : collect(Float64, saveat)
    m = twin.model
    n = nu(twin)
    Z = lock(() -> draw(rng, twin.belief, nsamples), twin.lock)
    now, callback, tstops = intervention_callbacks(m, collect(interventions), t0)

    S = Array{Float64, 3}(undef, n, length(ts), nsamples)
    tforeach(nsamples) do k
        z = @view Z[:, k]
        u = z[1:n]
        p = deepcopy(member_params(twin, z))
        for iv in now
            apply_now!(m, u, p, iv)
        end
        prob = member_problem(m, u, p, (t0, float(tend)))
        sol = solve_member(m, prob; saveat = ts, callback, tstops)
        SciMLBase.successful_retcode(sol) ||
            throw(ErrorException("forecast integration failed with retcode $(sol.retcode)"))
        for (j, uj) in enumerate(states_at(sol, ts))
            S[:, j, k] = uj
        end
    end
    return TwinForecast(String(name), ts, statenames(m), S)
end

"""
    scenario(twin, interventions, tend; name = "scenario", kwargs...)

Forecast the twin under a list of [`AbstractIntervention`](@ref)s (what-if
analysis). Keywords are passed to [`forecast`](@ref).

```julia
lockdown = scenario(twin, [SetParameter(30.0, :β => 0.1)], 120.0; name = "lockdown")
```
"""
function scenario(twin::Twin, interventions, tend::Real; name::AbstractString = "scenario",
        kwargs...)
    interventions isa AbstractIntervention && (interventions = [interventions])
    forecast(twin, tend; interventions, name, kwargs...)
end

"""
    compare(forecasts...; var, t = nothing, quantiles = (0.05, 0.95))

Summarize state `var` at time `t` (default: the final time) for each forecast.
Returns a Tables.jl-compatible named tuple of columns `scenario`, `mean`, `std`,
`lower` and `upper`.
"""
function compare(fcs::TwinForecast...; var, t = nothing, quantiles = (0.05, 0.95))
    rows = map(fcs) do fc
        j = t === nothing ? length(fc.t) : argmin(abs.(fc.t .- t))
        x = fc[var][j, :]
        (scenario = fc.name, mean = mean(x), std = std(x),
            lower = quantile(x, quantiles[1]), upper = quantile(x, quantiles[2]))
    end
    return Tables.columntable(collect(rows))
end
