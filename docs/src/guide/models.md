```@meta
CurrentModule = Vardr
```

# Models and sensors

A twin needs two descriptions of the real system:

- a **process model**, the [`TwinModel`](@ref): how the state evolves, `du/dt = f(u, p, t)`;
- a **measurement model**, the [`ObservationModel`](@ref): what the sensors see,
  `y = h(u, p, t) + ε` with `ε ~ N(0, R)`.

Together they form a state-space model; see
[The state-space model](@ref) for the theory.

```@setup models
using Vardr, LinearAlgebra, Statistics
```

## Process models

### From a plain Julia function

Write the right-hand side out-of-place, `f(u, p, t) -> du`, or in-place,
`f!(du, u, p, t)`. In-place is faster for larger systems.

```@example models
lv(u, p, t) = [p[1] * u[1] - p[2] * u[1] * u[2],     # prey
               p[3] * u[1] * u[2] - p[4] * u[2]]      # predator

model = TwinModel(lv, [10.0, 5.0], [1.1, 0.4, 0.1, 0.4];
    states = [:prey, :predator], params = [:α, :β, :δ, :γ])
```

- `u0` and `p` are the **nominal** values. The twin's initial belief is centred on them
  unless you give an explicit `prior`.
- `states` and `params` are optional, but names make every other call more readable
  (`twin[:prey]`, `SetParameter(t, :β => 0.3)`). Without names, use indices:
  `twin[1]`, `SetParameter(t, 2 => 0.3)`.
- `t0` sets the model's start time (default `0.0`).
- `p` can be any vector. Parameters are read and written through
  SymbolicIndexingInterface, so they keep their position in `p`.

### From ModelingToolkit

Any `ODEProblem` works, including one built with ModelingToolkit. States and
parameters are then referred to by their symbolic variables, or by `Symbol`:

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters β γ
@variables S(t) I(t) R(t)
@mtkcompile sir = System([D(S) ~ -β * S * I,
                          D(I) ~ β * S * I - γ * I,
                          D(R) ~ γ * I], t)
prob = ODEProblem(sir, [S => 0.99, I => 0.01, R => 0.0, β => 0.5, γ => 0.1], (0.0, 1.0))
model = TwinModel(prob)

sensors = ObservationModel(model, [I], 0.002)
twin = Twin(model, sensors; estimate = [β])
twin[I] == twin[:I]                       # both forms work
```

The `tspan` of the problem is irrelevant: only its start time is used. The twin
chooses the integration interval at each step.

!!! note "Initialization"
    ModelingToolkit problems normally run an initialization system that recomputes
    `u0` whenever parameters change. A twin's members already carry consistent states,
    so `TwinModel` disables this (`initializealg = NoInit()`). If your model has
    algebraic constraints (a DAE), make sure your prior respects them.

### Solver and solver options

The default solver is `Tsit5()`, which is good for non-stiff problems. Pass any other
SciML algorithm and solve keywords:

```julia
using OrdinaryDiffEqRosenbrock
model = TwinModel(prob; alg = Rodas5P(), abstol = 1e-8, reltol = 1e-6)
```

Use a stiff solver for chemical kinetics, detailed thermal networks or other models
with widely separated time scales. If an integration fails, [`predict!`](@ref) throws,
and the belief is left unchanged.

### Time-varying inputs

Known inputs (weather, a control schedule, pump demand) are functions of `t` inside
`f`:

```julia
outdoor(t) = 5.0 + 4.0 * sin(2π * (t - 9) / 24)
heater(t)  = 6 <= mod(t, 24) < 22 ? 1.0 : 0.0
room(u, p, t) = [((outdoor(t) - u[1]) / p[1] + p[2] * heater(t)) / p[3]]
model = TwinModel(room, [18.0], [5.0, 2.0, 0.5]; states = [:T], params = [:R, :P, :C],
    tstops = sort(vcat(6:24:240, 22:24:240)))   # stop at the switching times
```

When an input jumps, add the jump times as `tstops`, so the solver does not step over
them.

Inputs that only become known live, such as an actuator setting changed by an operator,
can be read from a log that your input feed appends to. See
[Inputs that arrive live](@ref).

### Discrete events: callbacks

Callbacks and `tstops` given to `TwinModel` are applied to every integration of every
member, and also in forecasts and calibration. They are combined with Vardr's own
callbacks, never replaced by them. A typical use is a dosing schedule:

```julia
using SciMLBase: DiscreteCallback
dose_times = 0.0:12.0:240.0
dosing = DiscreteCallback((u, t, integ) -> t in dose_times,
    integ -> (integ.u[1] += integ.p[8]))        # p[8] = dose amount
model = TwinModel(pk, u0, p; states, params, callback = dosing, tstops = dose_times)
```

See `examples/pkpd_patient_twin.jl` for a complete version.

### Large and spatial models

For a discretized PDE, write an in-place `f!` over the grid and name the cells. For
example, `states = [Symbol(:T, i) for i in 1:40]`. The twin works the same way. With
many unknowns, see [Localization](@ref guide-localization) in the estimation guide.

## Measurement models

### Sensors that read states directly

The most common case: give the states that are measured and their noise standard
deviations (a scalar or one per sensor).

```@example models
sensors = ObservationModel(model, [:prey, :predator], [0.5, 0.3])
sensors.names, sensors.R
```

### A custom measurement function

When the sensors measure a function of the state, pass `h(u, p, t)` and the noise.
The noise can be a vector of standard deviations or a full covariance matrix `R` (for
correlated sensor errors).

```@example models
# total population, and the predator fraction
h(u, p, t) = [u[1] + u[2], u[2] / (u[1] + u[2])]
sensors = ObservationModel(h, [1.0, 0.02]; names = [:total, :predator_fraction])

# correlated errors
R = [1.0 0.01; 0.01 0.02^2]
sensors = ObservationModel(h, R; names = [:total, :predator_fraction])
nothing # hide
```

`h` receives each member's parameters, including any that are being estimated. A
sensor may therefore depend on unknown parameters, such as a calibration gain or a
drug's potency.

!!! tip "Keep `h` cheap and pure"
    `h` is evaluated once per member per observation. Do not mutate `u` or `p` inside it.

### Missing values

An [`Observation`](@ref) always has one entry per output, in the order of the
observation model. Mark unavailable readings with `NaN`; they are skipped in the
update:

```julia
Observation(12.0, [3.4, NaN])     # only the first sensor reported
```

An observation where every value is `NaN` is a no-op. This makes it easy to merge
sensors that report at different rates into one stream.

### Units and scaling

Vardr does not rescale anything. Keep states, parameters and measurements in units
where their magnitudes are not wildly different, for example hours instead of seconds,
or mg/L instead of g/m³. This helps both the ODE solver and the estimator's
covariance computations.

## Simulation-only twins

A twin without an observation model is a stochastic simulator. It is useful for
Monte Carlo studies, or to generate synthetic "real" data:

```@example models
sim = Twin(model; estimator = NoAssimilation(N = 50))   # 10% spread around nominal values
fc = forecast(sim, 20.0)
round.(mean(fc)[:, end]; digits = 2)
```
