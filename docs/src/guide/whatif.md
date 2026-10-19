```@meta
CurrentModule = Vardr
```

# Forecasts and what-if scenarios

Once a twin is synchronized with the real system, it can answer two kinds of questions
without touching the real system:

- **"What happens next?"**: a [`forecast`](@ref) from the current belief.
- **"What if we do X?"**: a [`scenario`](@ref), which is a forecast with
  interventions.

Neither changes the twin: they work on samples drawn from its belief, so you can call
them at any time, even while the twin runs live.

```@setup whatif
using Vardr, Distributions, LinearAlgebra, Random, Statistics, DataFrames
osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]
real_system = TwinModel(osc, [1.5, 0.0], [3.0, 0.1])
truth(t) = Vardr.propagate(real_system, [1.5, 0.0], [3.0, 0.1], 0.0, t)
rng = Xoshiro(42)
measurements = [Observation(t, truth(t)[1] + 0.05 * randn(rng)) for t in 0.1:0.1:10.0]
model = TwinModel(osc, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
twin = Twin(model, ObservationModel(model, [:x], 0.05); estimator = EnKF(N = 100),
    estimate = [:k], prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2)),
    process_noise = 0.01, rng = Xoshiro(1))
foreach(o -> step!(twin, o), measurements)
```

The examples below use the oscillator twin from the [Tutorial](@ref "Tutorial: your
first twin"), synchronized up to `t = 10`.

## Forecasts

```julia
forecast(twin, tend; saveat, nsamples, interventions = [], name = "forecast", rng)
```

- `tend`: forecast end time (must be after `twin.t`).
- `saveat`: output times (default: 101 points from `twin.t` to `tend`).
- `nsamples`: number of trajectories (default: the number of members). For a particle
  filter, samples are drawn according to the weights.

Each sample starts from one member and keeps that member's estimated parameters, so the
forecast spread includes **both** state and parameter uncertainty. Process noise is not
added, so forecasts show the uncertainty in what the twin *knows*, not future random
disturbances.

```@example whatif
fc = forecast(twin, 15.0; saveat = 10.0:0.25:15.0)
```

### Working with a [`TwinForecast`](@ref)

| Call | Shape | Meaning |
|:---|:---|:---|
| `fc.t` | `ntimes` | output times |
| `fc.names` | `nstates` | state names |
| `mean(fc)`, `std(fc)` | `nstates × ntimes` | ensemble mean and standard deviation |
| `quantile(fc, q)` | `nstates × ntimes` | e.g. `q = 0.05`, `0.95` for a 90% band |
| `fc[:x]` | `ntimes × nsamples` | sample trajectories of one state |
| `fc.samples` | `nstates × ntimes × nsamples` | everything |

The sample trajectories turn any question into a probability. Write the condition per
sample, then average:

```@example whatif
X = fc[:x]
p_exceed = mean(any(abs.(X) .> 0.85; dims = 1))      # P(|x| > 0.85 at some time)
t_first_peak = [fc.t[argmax(X[:, k])] for k in axes(X, 2)]
(p_exceed = p_exceed, peak_time = mean(t_first_peak), peak_time_std = std(t_first_peak))
```

To plot a forecast band with any plotting package:

```julia
using CairoMakie
lo, hi = quantile(fc, 0.05)[1, :], quantile(fc, 0.95)[1, :]
band(fc.t, lo, hi; alpha = 0.3); lines!(fc.t, mean(fc)[1, :])
```

## Interventions

| Intervention | Effect at time `t` |
|:---|:---|
| [`SetParameter`](@ref)`(t, :c => 1.0, :k => 2.5)` | change one or more parameters |
| [`SetState`](@ref)`(t, :v => 0.0)` | overwrite one or more states |
| [`Event`](@ref)`(t, integ -> ...)` | anything else: an arbitrary `affect!(integrator)` |

- An intervention at or before the forecast start is applied immediately.
- `SetParameter` on an **estimated** parameter overrides every sample's value, so its
  uncertainty is gone after `t`.
- Combine any number of interventions in a vector. They are applied in time order.

```@example whatif
plans = [
    forecast(twin, 20.0; name = "do nothing"),
    scenario(twin, SetParameter(12.0, :c => 1.0), 20.0; name = "add damper at 12"),
    scenario(twin, [SetState(11.0, :v => 0.0), SetParameter(11.0, :c => 0.5)], 20.0;
        name = "brake at 11"),
    scenario(twin, Event(12.0, integ -> (integ.u[1] *= 0.5)), 20.0;
        name = "halve x at 12"),
]
DataFrame(compare(plans...; var = :x, t = 15.0))
```

`Event` gives full access to the SciML integrator: `integ.u`, `integ.p`, `integ.t`. Use
it for a pulse input, a dose, or a valve that closes. Modify `integ.u` and `integ.p`
in place.

### Recurring actions

For actions that repeat (a dosing regimen, a daily schedule), generate the list:

```julia
regimen(amount) = [Event(t, integ -> (integ.u[1] += amount)) for t in 12.0:12.0:72.0]
scenario(twin, regimen(150.0), 72.0; name = "150 mg q12h")
```

If the intervention changes an input that the model reads from a parameter, a sequence
of `SetParameter`s is simpler.

## Comparing options

[`compare`](@ref) summarizes one state at one time across forecasts. It returns a
Tables.jl table with the columns `scenario`, `mean`, `std`, `lower` and `upper`:

```julia
compare(fcs...; var = :x, t = nothing, quantiles = (0.05, 0.95))
```

Real decisions usually need a custom metric that combines the whole trajectory: total
throughput, time in a target range, or risk of exceeding a limit. Compute it per sample
from `fc[...]`, then summarize:

```@example whatif
# time (per sample) during which |x| < 0.2
time_calm(fc) = vec(sum(abs.(fc[:x]) .< 0.2; dims = 1)) .* (fc.t[2] - fc.t[1])
DataFrame(option = [fc.name for fc in plans],
          calm_time = [mean(time_calm(fc)) for fc in plans],
          P_calm_over_3 = [mean(time_calm(fc) .> 3) for fc in plans])
```

The examples `pkpd_patient_twin.jl` (dosing regimens: efficacy against toxicity) and
`tank_leak_detection_twin.jl` (throughput against overflow risk) show this pattern in
full.

## Good practice

- **Use the same `nsamples` and a fixed `rng`** for every option in a comparison. The
  differences then reflect the interventions, not sampling noise. With the default
  `nsamples` (the number of members), an EnKF forecast uses every member exactly once.
- **Forecast skill is limited.** Uncertainty grows with the horizon, much faster in
  chaotic systems (see `examples/lorenz63_twin.jl`). Report the spread, not just the
  mean.
- **Model what can change.** A scenario can only change what the model represents. If
  an option acts on something that is hard-coded, make it a parameter.
