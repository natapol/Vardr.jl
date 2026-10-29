```@meta
CurrentModule = Vardr
```

# Models and sensors

A twin needs two descriptions of the real system:

- a **process model** (an [`AbstractTwinModel`](@ref)): how the state `u` evolves;
- a **measurement model**, the [`ObservationModel`](@ref): what the sensors see,
  `y = h(u, p, t) + ε` with `ε ~ N(0, R)`.

Together they form a state-space model; see
[The state-space model](@ref) for the theory.

```@setup models
using Vardr, LinearAlgebra, Statistics, Random, Distributions
```

## Choosing a model type

| Model type | Use it when | State `u` | Typical estimator |
|:--|:--|:--|:--|
| [`ODEModel`](@ref) | The mechanisms are known as equations: `du/dt = f(u, p, t)` | The ODE state | [`EnKF`](@ref) |
| [`NeuralODEModel`](@ref) | Part of the dynamics is unknown, but data are available to learn it | The ODE state | [`EnKF`](@ref) |
| [`AgentModel`](@ref) | Behaviour emerges from many interacting individuals, or individual detail matters (contact patterns, fixed durations, heterogeneity) | Summary counts of the agents | [`ParticleFilter`](@ref) |

All three share the rest of Vardr: observation models, the filters, forecasts,
scenarios, calibration, streaming and snapshots.

## ODE models

### From a plain Julia function

Write the right-hand side out-of-place, `f(u, p, t) -> du`, or in-place,
`f!(du, u, p, t)`. In-place is faster for larger systems.

```@example models
lv(u, p, t) = [p[1] * u[1] - p[2] * u[1] * u[2],     # prey
               p[3] * u[1] * u[2] - p[4] * u[2]]      # predator

model = ODEModel(lv, [10.0, 5.0], [1.1, 0.4, 0.1, 0.4];
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
model = ODEModel(prob)

sensors = ObservationModel(model, [I], 0.002)
twin = Twin(model, sensors; estimate = [β])
twin[I] == twin[:I]                       # both forms work
```

The `tspan` of the problem is irrelevant: only its start time is used. The twin
chooses the integration interval at each step.

!!! note "Initialization"
    ModelingToolkit problems normally run an initialization system that recomputes
    `u0` whenever parameters change. A twin's members already carry consistent states,
    so `ODEModel` disables this (`initializealg = NoInit()`). If your model has
    algebraic constraints (a DAE), make sure your prior respects them.

### Solver and solver options

The default solver is `Tsit5()`, which is good for non-stiff problems. Pass any other
SciML algorithm and solve keywords:

```julia
using OrdinaryDiffEqRosenbrock
model = ODEModel(prob; alg = Rodas5P(), abstol = 1e-8, reltol = 1e-6)
```

Use a stiff solver for chemical kinetics, detailed thermal networks or other models
with widely separated time scales. If an integration fails, [`predict!`](@ref) throws an
[`IntegrationError`](@ref), and the belief is left unchanged.

### Time-varying inputs

Known inputs (weather, a control schedule, pump demand) are functions of `t` inside
`f`:

```julia
outdoor(t) = 5.0 + 4.0 * sin(2π * (t - 9) / 24)
heater(t)  = 6 <= mod(t, 24) < 22 ? 1.0 : 0.0
room(u, p, t) = [((outdoor(t) - u[1]) / p[1] + p[2] * heater(t)) / p[3]]
model = ODEModel(room, [18.0], [5.0, 2.0, 0.5]; states = [:T], params = [:R, :P, :C],
    tstops = sort(vcat(6:24:240, 22:24:240)))   # stop at the switching times
```

When an input jumps, add the jump times as `tstops`, so the solver does not step over
them.

Inputs that only become known live, such as an actuator setting changed by an operator,
can be read from a log that your input feed appends to. See
[Inputs that arrive live](@ref).

### Discrete events: callbacks

Callbacks and `tstops` given to `ODEModel` are applied to every integration of every
member, and also in forecasts and calibration. They are combined with Vardr's own
callbacks, never replaced by them. A typical use is a dosing schedule:

```julia
using SciMLBase: DiscreteCallback
dose_times = 0.0:12.0:240.0
dosing = DiscreteCallback((u, t, integ) -> t in dose_times,
    integ -> (integ.u[1] += integ.p[8]))        # p[8] = dose amount
model = ODEModel(pk, u0, p; states, params, callback = dosing, tstops = dose_times)
```

See `examples/pkpd_patient_twin.jl` for a complete version.

### Large and spatial models

For a discretized PDE, write an in-place `f!` over the grid and name the cells. For
example, `states = [Symbol(:T, i) for i in 1:40]`. The twin works the same way. With
many unknowns, see [Localization](@ref guide-localization) in the estimation guide.

## [Neural and hybrid ODE models](@id guide-neural)

A [`NeuralODEModel`](@ref) is an ODE whose right-hand side is, or contains, a neural
network.

- A **neural ODE** learns all of the dynamics from data: `du/dt = NN(u; w)`
  [ChenEtAl2018](@cite).
- A **hybrid model**, or universal differential equation, keeps what is known (mass
  balances, conservation laws) and lets the network learn only the unknown part, such as
  a reaction rate [PsichogiosUngar1992, RackauckasEtAl2020](@cite). Hybrid models need
  far less data than a pure network and extrapolate better, because the known structure
  still holds outside the training data.

The network is any callable `network(x, w)` with a flat weight vector `w`. Vardr ships
a small multilayer perceptron, [`MLP`](@ref), which needs no deep-learning package and
works with forward-mode automatic differentiation.

As an example, take logistic growth in which the crowding term is unknown. The known
part is that the population grows at rate `r` when it is small. The network learns how
growth slows down:

```@example models
using ForwardDiff
using Optimization: AutoForwardDiff
using OptimizationOptimJL: BFGS

# Data from the real system: dN/dt = r N (1 - N/K), with K = 10 unknown to the modeller
logistic(u, p, t) = [p[1] * u[1] * (1 - u[1] / 10)]
real_system = ODEModel(logistic, [0.5], [1.0])
data = [Observation(t, Vardr.propagate(real_system, [0.5], [1.0], 0.0, t)[1] + 0.05randn(Xoshiro(round(Int, 10t))))
        for t in 0.5:0.5:8.0]

# Hybrid model: dN/dt = r N (1 + NN(N/10))
crowding = MLP(1, 6, 1)
ude = NeuralODEModel(crowding, [0.5]; p = [1.0], params = [:r], states = [:N],
    f = (u, p, t, nn) -> [p[1] * u[1] * (1 + nn([u[1] / 10])[1])],
    rng = Xoshiro(3), maxiters = 10_000)
```

`f(u, p, t, nn)` receives the mechanistic parameters `p` and the network as a function
`nn(x)`. The weights are parameters named `nn1, nn2, …` ([`weightnames`](@ref)), so
training is ordinary [`calibrate!`](@ref) with gradients:

```@example models
trainer = Twin(ude, ObservationModel(ude, [:N], 0.05);
    estimator = NoAssimilation(N = 1), prior = fill(0.5, 1, 1))
fit = calibrate!(trainer, data, weightnames(ude);
    alg = BFGS(initial_stepnorm = 0.01), adtype = AutoForwardDiff(), u0 = [0.5])
w = networkweights(ude, trainer.p)
[(N = N, learned = crowding([N / 10], w)[1], true_term = -N / 10) for N in (1, 5, 9)]
```

The trained weights are now the twin's parameters (`trainer.p`). For a twin of a new
system, build the model with `weights = w`, or keep using `trainer`.

!!! tip "Training neural and hybrid models"
    - **Build in what you know.** Bound or sign-constrain the network output when the
      physics demands it, for example `rate = 1 / (1 + exp(-nn(x)[1]))` for a rate in
      ``(0, 1)``. Unconstrained networks can make the ODE blow up during training.
    - **Scale inputs** to order one (`nn([S / 10])`).
    - **Start gently.** Gradient-based optimizers such as `BFGS` take a large first step
      when the initial misfit is large; `BFGS(initial_stepnorm = 0.01)` avoids leaping
      into unstable weights. Pass `maxiters` to the model so that a diverging trajectory
      fails fast.
    - **Cover the operating range.** A network is only trustworthy where it saw data;
      check the learned function before you rely on it.

### Online learning

Network weights can also be estimated online, like any parameter. Two common patterns:

- **Fine-tune a few weights.** Train offline, then estimate only the output bias (the
  last weight) online, so that the twin can shift the learned function for the system
  in front of it. `examples/hybrid_bioreactor_twin.jl` does this.
- **Estimate all weights** of a small network with the EnKF
  (`estimate = weightnames(ude)`). This is ensemble Kalman inversion
  [IglesiasLawStuart2013, KovachkiStuart2019](@cite); it needs no gradients but works
  best with tens, not thousands, of weights.

### Other neural network libraries

Any network whose parameters can be flattened into a vector works. With Lux.jl and
ComponentArrays.jl, for example:

```julia
using Lux, ComponentArrays
chain = Chain(Dense(2 => 16, tanh), Dense(16 => 2))
ps, st = Lux.setup(Xoshiro(0), chain)
ps = ComponentArray(ps)
network(x, w) = first(chain(x, ComponentArray(w, getaxes(ps)), st))
model = NeuralODEModel(network, u0; weights = collect(ps))
```

## [Agent-based models](@id guide-agents)

An [`AgentModel`](@ref) is a stochastic simulation of individual agents, stepped in
fixed time steps `dt`. You provide three functions:

- `init(u, p, rng) -> sim`: build a simulation (the agents and their environment);
- `step!(sim, p, t, rng)`: advance it one step, drawing all randomness from `rng`;
- `summarize(sim) -> u`: the *summary state*, a short vector such as compartment
  counts.

Each member of the twin's belief owns a whole simulation. The summary `u` is what the
twin's history, forecasts and observation model see. The simulation itself is the
member's *latent* state, in `twin.belief.latent`.

```@example models
mutable struct Village
    status::Vector{Int8}      # 0 susceptible, 1 infectious, 2 recovered
    new_cases::Int
end

function village(u, p, rng)                      # u: summary drawn from the prior
    status = zeros(Int8, 1000)
    status[1:clamp(round(Int, u[2]), 1, 1000)] .= 1
    Village(status, 0)
end

function village_step!(v, p, t, rng)
    β, γ = p
    p_inf = 1 - exp(-β * count(==(1), v.status) / length(v.status))
    v.new_cases = 0
    for i in eachindex(v.status)
        if v.status[i] == 0 && rand(rng) < p_inf
            v.status[i] = 1
            v.new_cases += 1
        elseif v.status[i] == 1 && rand(rng) < γ
            v.status[i] = 2
        end
    end
end

counts(v) = [count(==(0), v.status), count(==(1), v.status), count(==(2), v.status),
    v.new_cases]

abm = AgentModel(village, village_step!, counts, [990, 10, 0, 0], [0.5, 0.2];
    states = [:S, :I, :R, :new], params = [:β, :γ], dt = 1.0)
```

The particle filter is the natural estimator. It only reweights members and copies
whole simulations, so it never has to edit agents. Here 60% of new cases are reported
each day, and the twin learns the transmission rate `β`:

```@example models
reported = ObservationModel((u, p, t) -> [0.6 * u[4]], [2.0]; names = [:reported])
abm_twin = Twin(abm, reported; estimator = ParticleFilter(N = 200, jitter = 0.3),
    estimate = [:β], prior = MvNormal([990, 10, 0, 0, 0.4], Diagonal([1, 3, 1, 1, 0.15] .^ 2)),
    process_noise = [0, 0, 0, 0, 0.01], rng = Xoshiro(1))

real_village = village([990, 10, 0, 0], [0.6, 0.2], Xoshiro(2))
rng = Xoshiro(3)
for day in 1:20
    village_step!(real_village, [0.6, 0.2], day - 1, rng)
    step!(abm_twin, Observation(day, 0.6 * real_village.new_cases + 2randn(rng)))
end
(β = abm_twin[:β], β_sd = uncertainty(abm_twin, :β), I = abm_twin[:I], I_true = counts(real_village)[2])
```

Forecasts and scenarios simulate copies of the members' agents. An [`Event`](@ref)
acts on the agents directly; its function is called as `affect!(sim, p, t)`:

```@example models
vaccinate = Event(22.0, (sim, p, t) -> replace!(sim.status, 0 => 2))
fc = scenario(abm_twin, vaccinate, 40.0; saveat = [40.0])
mean(fc[:S][end, :])
```

Rules that follow from the summary state being computed from the agents:

- **Noise and bounds** apply only to estimated parameters. The simulation supplies its
  own randomness, so `process_noise` and `bounds` on the summary states are rejected.
- **The EnKF** shifts each member's state, which agents cannot follow on their own.
  Give the model a `lift(sim, u, p, rng)` function that edits the agents to match an
  updated summary (for example by moving agents between compartments), or use the
  particle filter.
- **Time** advances in whole steps: observation and forecast times must lie on the
  `dt` grid. Interventions take effect at the first step at or after their time.
  [`SetState`](@ref) is not available; use an `Event`.
- **Randomness** comes from a fresh generator per member per prediction, seeded from the
  twin's `rng`. Runs are reproducible however many threads you use, and members that
  the particle filter duplicates evolve independently.
- **Static parameters** suffer from sample impoverishment: after a few resamplings
  only a handful of distinct parameter values survive, and the filter becomes
  overconfident. Keep them diverse with `jitter` and a little `process_noise` on the
  parameter. In the example above, `jitter = 0.1` with noise `0.005` gave z-scores with
  an RMS of 2.8 over ten simulated villages (overconfident). `jitter = 0.3` with noise
  `0.01` gave 0.97 (calibrated).
- **Calibration** with [`calibrate!`](@ref) simulates `nsims` replicates with fixed
  seeds and fits their mean. It is a quick starting point; the particle filter's
  online estimate accounts for the model's randomness properly.
- **Cost** grows with members × agents × steps. Members run in parallel across
  threads; keep the population as small as the question allows.

Particle filters for agent-based models are an active research area
[WardEvansMalleson2016, MallesonEtAl2020](@cite). `examples/agent_epidemic_twin.jl`
is a larger example with age groups, two data streams and interventions on contacts.

### Using Agents.jl

Models written with [Agents.jl](https://juliadynamics.github.io/Agents.jl/) plug in
through the same three functions. Reseed the model's own generator from `rng` in
every step, so that copied members do not share random streams:

```julia
using Agents
init(u, p, rng) = my_abm(; β = p[1], seed = rand(rng, UInt))   # your model constructor
function abm_step!(model, p, t, rng)
    abmproperties(model)[:β] = p[1]           # pass (estimated) parameters in
    Random.seed!(abmrng(model), rand(rng, UInt))
    Agents.step!(model, 1)
end
summarize(model) = [count(a -> a.status == :S, allagents(model)),
                    count(a -> a.status == :I, allagents(model))]
abm = AgentModel(init, abm_step!, summarize, [990, 10], [0.5]; states = [:S, :I], params = [:β])
```

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
