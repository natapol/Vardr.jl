```@meta
CurrentModule = Vardr
```

# Tutorial: your first twin

This tutorial builds a complete digital twin step by step. The system is a damped
oscillator (a mass on a spring) whose spring constant we do not know exactly:

```math
\ddot x = -k\,x - c\,\dot x
```

We measure its position `x` with a noisy sensor. By the end, the twin will:

1. track the full state (`x` and the unmeasured velocity `v`) from noisy measurements,
2. learn the spring constant `k` while it runs,
3. forecast the future with uncertainty, and
4. answer a what-if question.

Every code block on this page runs when the documentation is built.

## 0. Setup

```@example tutorial
using Vardr, Distributions, LinearAlgebra, Random, Statistics
```

Start Julia with threads (`julia -t auto`): the twin integrates its ensemble members in
parallel.

## 1. Some "real" data

In practice the data comes from your system. Here we simulate a real oscillator with
`k = 3.0` and add sensor noise (σ = 0.05). [`Vardr.propagate`](@ref) integrates a
model from one time to another; we use it to create the ground truth.

```@example tutorial
osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]    # u = [x, v], p = [k, c]

real_system = ODEModel(osc, [1.5, 0.0], [3.0, 0.1])
truth(t) = Vardr.propagate(real_system, [1.5, 0.0], [3.0, 0.1], 0.0, t)

rng = Xoshiro(42)
measurements = [Observation(t, truth(t)[1] + 0.05 * randn(rng)) for t in 0.1:0.1:10.0]
first(measurements, 3)
```

An [`Observation`](@ref) is just a time and a vector of measured values.

## 2. The model

An [`ODEModel`](@ref) wraps the equations, the nominal initial state and the parameters.
Naming states and parameters lets you refer to them as `:x` or `:k` everywhere.
Our design value of the spring constant is `k = 4.0`, which is wrong.

```@example tutorial
model = ODEModel(osc, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
statenames(model), paramnames(model)
```

## 3. The sensors

An [`ObservationModel`](@ref) describes what is measured and how noisy it is. Here one
sensor reads `x` with standard deviation 0.05.

```@example tutorial
sensors = ObservationModel(model, [:x], 0.05)
nothing # hide
```

## 4. The twin

The [`Twin`](@ref) combines the model and sensors with a *belief*: an ensemble of
possible states of the real system. Its key choices are:

- `estimator`: how to update the belief when data arrives. The [`EnKF`](@ref) (ensemble
  Kalman filter) is a robust default.
- `estimate`: parameters to learn online. They are appended to the state, so the twin
  works with `z = [x, v, k]`.
- `prior`: what we believe before seeing data, over all of `z`.
- `process_noise`: how much the real system may drift away from the model per unit
  time. A small value on `k` keeps the twin able to learn.

```@example tutorial
prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2))  # over [x, v, k]
twin = Twin(model, sensors; name = "oscillator",
    estimator = EnKF(N = 100), estimate = [:k], prior,
    process_noise = [0.01, 0.01, 0.01], rng = Xoshiro(1))
```

## 5. Synchronize with data

[`step!`](@ref) advances the twin to the observation's time ([`predict!`](@ref)) and
then corrects it with the measurement ([`assimilate!`](@ref)):

```@example tutorial
for obs in measurements
    step!(twin, obs)
end
twin
```

Query the twin by name. `twin[:x]` is the posterior mean and [`uncertainty`](@ref) the
posterior standard deviation:

```@example tutorial
(x = twin[:x], x_true = truth(twin.t)[1], v = twin[:v], v_true = truth(twin.t)[2])
```

```@example tutorial
(k = twin[:k], k_std = uncertainty(twin, :k), k_true = 3.0)
```

The twin has learned `k` and the velocity, even though neither is measured.
[`paramvalues`](@ref) lists all parameters, both estimated and fixed:

```@example tutorial
paramvalues(twin)
```

## 6. Look at the history

Every prediction and update is recorded. [`history`](@ref) returns a Tables.jl table,
so it converts directly to a `DataFrame`:

```@example tutorial
using DataFrames
df = DataFrame(history(twin))
last(df[df.kind .== :update, [:t, :x, :x_std, :k, :k_std]], 5)
```

## 7. Forecast

[`forecast`](@ref) propagates samples of the current belief into the future without
changing the twin. Parameter uncertainty is carried along.

```@example tutorial
fc = forecast(twin, 20.0; saveat = 10.0:0.5:20.0)
```

```@example tutorial
x_mean = mean(fc)[1, :]                 # states × times
x_lo, x_hi = quantile(fc, 0.05)[1, :], quantile(fc, 0.95)[1, :]
DataFrame(t = fc.t, mean = x_mean, lower = x_lo, upper = x_hi)[1:4:end, :]
```

`fc[:x]` gives the raw sample trajectories (`times × samples`), which you can use to
compute any probability you like. For example, the chance that `|x|` exceeds 0.85 at
some point in the forecast:

```@example tutorial
mean(maximum(abs.(fc[:x]); dims = 1) .> 0.85)
```

## 8. Ask "what if?"

A [`scenario`](@ref) is a forecast with interventions. What if we add a damper
(`c → 1.0`) at `t = 12`?

```@example tutorial
baseline = forecast(twin, 20.0)
damped = scenario(twin, [SetParameter(12.0, :c => 1.0)], 20.0; name = "damper at t = 12")
DataFrame(compare(baseline, damped; var = :x))
```

## 9. Save and resume

```@example tutorial
path = snapshot(joinpath(mktempdir(), "oscillator.jld2"), twin)

twin2 = Twin(model, sensors; estimator = EnKF(N = 100), estimate = [:k], prior,
    process_noise = [0.01, 0.01, 0.01])
restore!(twin2, path)
twin2[:k] == twin[:k]
```

## Where next

- [Step-by-step workflow](@ref): the method for building a twin you can trust, from
  the question to deployment, on a complete worked case.
- Theory: [Digital twins and state-space models](@ref),
  [Bayesian filtering and data assimilation](@ref), and
  [Prediction, calibration and verification](@ref).

- [Models and sensors](@ref): ModelingToolkit models, inputs, callbacks, and custom
  measurement functions.
- [State and parameter estimation](@ref): choosing and tuning estimators.
- [Forecasts and what-if scenarios](@ref): interventions and decision analysis.
- [Calibration and persistence](@ref): offline fitting and snapshots.
- [Running live](@ref): streaming data, sinks and outages.
- [Troubleshooting](@ref): what to do when the twin goes wrong.
