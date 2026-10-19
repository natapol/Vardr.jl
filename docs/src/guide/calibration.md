```@meta
CurrentModule = Vardr
```

# Calibration and persistence

```@setup calib
using Vardr, Distributions, LinearAlgebra, Random, Statistics
osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]
real_system = TwinModel(osc, [1.5, 0.0], [3.0, 0.1])
truth(t) = Vardr.propagate(real_system, [1.5, 0.0], [3.0, 0.1], 0.0, t)
rng = Xoshiro(42)
measurements = [Observation(t, truth(t)[1] + 0.05 * randn(rng)) for t in 0.1:0.1:10.0]
model = TwinModel(osc, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
sensors = ObservationModel(model, [:x], 0.05)
```

## Offline calibration vs online estimation

Vardr offers two ways to learn parameters:

| | [`calibrate!`](@ref) (offline) | `estimate = [...]` (online) |
|:---|:---|:---|
| Data | a batch of historical observations | one observation at a time |
| Method | maximum likelihood, by optimization | Bayesian filtering |
| Result | point estimates, written into the twin's fixed parameters | a posterior that evolves |
| Use for | commissioning: fitting a new twin to historical data | following a system as it runs and drifts |

They work well together: calibrate most parameters once on history, then estimate the
few that drift online.

## `calibrate!`

```julia
calibrate!(twin, data, params; alg, lb = nothing, ub = nothing,
           adtype = SciMLBase.NoAD(), u0 = state(twin), solve_kwargs...)
```

It simulates the model from `u0` at `twin.t` through all `data`, and minimizes the
Gaussian negative log-likelihood under the twin's observation model. Missing (`NaN`)
values are ignored.

```@example calib
using OptimizationOptimJL: NelderMead

twin = Twin(model, sensors; estimator = EnKF(N = 50), rng = Xoshiro(1))
sol = calibrate!(twin, measurements, [:k, :c]; alg = NelderMead(), u0 = [1.5, 0.0])
sol.u, paramvalues(twin)
```

- `params`: parameter names (or indices) to fit; the others stay fixed.
- `alg`: any Optimization.jl optimizer. `NelderMead()` needs no gradients. For
  gradient-based optimizers (`LBFGS()`), pass `adtype = AutoForwardDiff()`, and make
  sure your model function accepts dual numbers (avoid `Float64[...]` literals inside
  `f`).
- `lb`, `ub`: bounds, for optimizers that support them.
- `u0`: the initial state of the simulation. It defaults to the twin's current mean
  state. If the initial state is uncertain too, include it in your own loss, or start
  the data a little later and let the filter handle it.
- Extra keywords (`maxiters = 500`) go to `Optimization.solve`.

After calibration, the fitted values become the twin's fixed parameters. If a calibrated
parameter is also in `estimate`, its members are shifted so that their mean equals the
fitted value, keeping their spread.

All observations must be at or after `twin.t`. Calibrate a fresh twin on history, then
assimilate newer data.

## Snapshots

A twin that runs for weeks must survive restarts. [`snapshot`](@ref) saves its dynamic
state to a JLD2 file: time, belief, fixed parameters, history and RNG state.

```@example calib
twin = Twin(model, sensors; estimator = EnKF(N = 50), estimate = [:k],
    process_noise = 0.01, rng = Xoshiro(1))
foreach(o -> step!(twin, o), measurements[1:50])
path = snapshot(joinpath(mktempdir(), "twin.jld2"), twin)
```

Code is **not** saved: the model function, observation function and estimator settings
belong in your source code. To restore, build the twin with the same code, then load the
state:

```@example calib
build() = Twin(model, sensors; estimator = EnKF(N = 50), estimate = [:k],
    process_noise = 0.01, rng = Xoshiro(0))
restored = restore!(build(), path)
foreach(o -> step!(restored, o), measurements[51:end])
foreach(o -> step!(twin, o), measurements[51:end])
restored[:k] == twin[:k]          # continues exactly where it left off
```

`restore!` checks that the snapshot fits the twin: the same state dimension and the same
belief type. With an explicit RNG of the same type (`Xoshiro`), the restored twin
continues bit-for-bit identically.

While running live, [`SnapshotSink`](@ref) writes snapshots periodically and at
shutdown. See [Running live](@ref).
