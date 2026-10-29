```@meta
CurrentModule = Vardr
```

# Step-by-step workflow

The [Tutorial](@ref "Tutorial: your first twin") shows the mechanics. This chapter shows
the **method**: the steps to go from a real-world question to a twin you can trust, with
the theory behind each step and a check that tells you whether to move on.

| Step | Question | Theory |
|:---|:---|:---|
| [1. Define the purpose](@ref wf-purpose) | What decision will the twin support? | [Digital twins](@ref "What is a digital twin?") |
| [2. Write the process model](@ref wf-model) | How does the state evolve? | [State-space model](@ref "The state-space model") |
| [3. Describe the sensors](@ref wf-sensors) | What is measured, how noisily? | [State-space model](@ref "The state-space model") |
| [4. Check identifiability](@ref wf-ident) | Can the data inform the unknowns? | [Identifiability](@ref) |
| [5. Build a synthetic truth](@ref wf-osse) | Can we test the twin against a known answer? | [Verifying a twin](@ref "Verifying a twin: is the uncertainty honest?") |
| [6. Configure the twin](@ref wf-twin) | Prior, estimator, model error, bounds | [Bayesian filtering](@ref "Bayesian filtering and data assimilation") |
| [7. Evaluate and tune](@ref wf-tune) | Is it accurate *and* honest about its uncertainty? | [Verifying a twin](@ref "Verifying a twin: is the uncertainty honest?") |
| [8. Replay as a live stream](@ref wf-replay) | Does it work in operation, with alerts? | [Sequential processing](@ref "Sequential processing of live data") |
| [9. Support the decision](@ref wf-decide) | What happens if we act, or don't? | [What-if scenarios](@ref "What-if scenarios as interventions") |
| [10. Deploy and maintain](@ref wf-deploy) | How do we keep it trustworthy? | [Innovations](@ref "Without ground truth (real data): innovations") |

All code on this page runs when the documentation is built.

```@example wf
using Vardr, Distributions, LinearAlgebra, Random, Statistics, DataFrames
```

## [1. Define the purpose](@id wf-purpose)

A twin is built for a decision, not for its own sake. Write down the question before
writing any equations, because it determines what must be modelled, measured and
predicted.

**Our case.** A process water tank drains through an outlet that slowly **clogs** with
deposits. As the outlet clogs, the level rises, and above 3 m the tank overflows. A
level sensor is installed, but the clogging cannot be observed directly. Operations
want to know:

1. How clogged is the outlet now, and how fast is it clogging?
2. When will the tank overflow if nothing is done?
3. Should we clean the outlet now, later, or reduce the inflow instead?

This tells us the twin must estimate an **unobserved state** (the outlet coefficient)
and an **unknown rate** (clogging speed), and must **forecast with uncertainty** about
24 h ahead.

!!! tip "Check before moving on"
    Can you name the quantities to estimate, the forecast horizon, and the actions to
    compare? If not, refine the question first.

## [2. Write the process model](@id wf-model)

Use the simplest physics that can answer the question. For a tank with cross-section
``A`` (m²), level ``h`` (m), a known inflow schedule ``q_\text{in}(t)`` (m³/h) and an
outlet with coefficient ``c`` (Torricelli's law, outflow ``c\sqrt h``):

```math
A\,\frac{\mathrm dh}{\mathrm dt} = s\,q_\text{in}(t) - c\sqrt{h},
\qquad
\frac{\mathrm dc}{\mathrm dt} = -r .
```

``c`` is modelled as a **state** that decreases at an unknown clogging rate ``r``, a
**parameter**. ``s`` scales the inflow, which lets us express the "reduce inflow" action
later. Time is in hours.

```@example wf
inflow(t) = 0.6 + 0.15 * sin(2π * t / 12)       # known production schedule (m³/h)

function tank(u, p, t)
    h, c = u
    A, r, s = p
    return [(s * inflow(t) - c * sqrt(max(h, 0.0))) / A,   # level
            -r]                                            # outlet coefficient
end

model = ODEModel(tank, [1.6, 0.45], [2.0, 0.0, 1.0];
    states = [:h, :c], params = [:A, :r, :s])
```

Sanity-check the model: the steady level is ``h^* = (q/c)^2``. A clean outlet
(``c = 0.5``) gives about 1.44 m, and the tank overflows once ``c`` falls below about
0.35.

```@example wf
(clean = (0.6 / 0.5)^2, c_at_overflow = 0.6 / sqrt(3.0))
```

!!! tip "Check before moving on"
    Simulate the model over the range of conditions you expect, and check the
    equilibria, signs and time scales. Use units that keep magnitudes moderate (hours,
    not seconds). Guard against invalid regions: `sqrt(max(h, 0))`.

## [3. Describe the sensors](@id wf-sensors)

A radar level sensor reads ``h`` with a standard deviation of 2 cm, every 15 minutes.
The outlet coefficient is not measured.

```@example wf
σ = 0.02
sensors = ObservationModel(model, [:h], σ)
nothing # hide
```

Take ``\sigma`` from the sensor datasheet, or from the scatter of readings while the
process is steady. It sets how much the twin trusts each reading.

!!! tip "Check before moving on"
    Every measured quantity must be computable from the model: here, directly from the
    state. If sensors measure a function of the state, write `h(u, p, t)` (see
    [Models and sensors](@ref)).

## [4. Check identifiability](@id wf-ident)

Before trying to estimate unknowns, check that the data *can* inform them
([Identifiability](@ref)). Perturb each unknown by its plausible uncertainty, simulate
the measured output, and compare the change with the sensor noise:

```@example wf
function simulate(u0, p, ts)
    u, t, out = copy(u0), 0.0, Float64[]
    for tk in ts
        u = Vardr.propagate(model, u, p, t, tk)
        t = tk
        push!(out, u[1])
    end
    return out
end

ts = 0.25:0.25:12.0
base = simulate([1.44, 0.5], [2.0, 0.0, 1.0], ts)
effect(u0, p) = maximum(abs.(simulate(u0, p, ts) .- base)) / σ   # in units of σ

DataFrame(unknown = ["c: -10%", "r: 0 → 0.004 /h", "A: +10%"],
    max_change_in_σ = [effect([1.44, 0.45], [2.0, 0.0, 1.0]),
                       effect([1.44, 0.5], [2.0, 0.004, 1.0]),
                       effect([1.44, 0.5], [2.2, 0.0, 1.0])])
```

Over 12 hours, ``c`` and ``r`` change the level by many times the noise, so they are
identifiable. A 10% change in ``A`` moves the level by only about one ``\sigma``: the
data says almost nothing about the tank's area. Fortunately it is known from the
drawings, so we keep it fixed.

!!! tip "Check before moving on"
    Estimate only what is uncertain **and** identifiable. Fix the rest from design data
    or a dedicated test ([`calibrate!`](@ref) on a commissioning experiment).

## [5. Build a synthetic truth](@id wf-osse)

Never tune a twin on real data first: you cannot tell whether it is right. Run an
*observing-system simulation experiment*: simulate a "real" tank with known behaviour,
make noisy measurements of it, and check that the twin recovers what we know is true.

To make the test honest, the synthetic truth must contain things **the model does not
know about**: inflow disturbances, and a sudden blockage by debris at 18 h. A twin is
only as good as its ability to cope with surprises.

```@example wf
disturbance(t) = 0.04 * sin(2π * t / 5.3) + 0.03 * sin(2π * t / 2.1)   # unmodelled
real_tank(u, p, t) = [(inflow(t) + disturbance(t) - u[2] * sqrt(max(u[1], 0.0))) / 2.0,
                      -0.003]                       # true clogging rate 0.003 /h
real_system = ODEModel(real_tank, [1.44, 0.5], Float64[])

tobs = 0.25:0.25:30.0
truth = Dict{Float64, Vector{Float64}}()
let u = [1.44, 0.5], t = 0.0
    for tk in tobs
        u = Vardr.propagate(real_system, u, Float64[], t, tk)
        t = tk
        tk == 18.0 && (u[2] -= 0.04)               # debris partly blocks the outlet
        truth[tk] = u
    end
end

rng = Xoshiro(11)
data = [Observation(t, truth[t][1] + σ * randn(rng)) for t in tobs]
(n_observations = length(data), true_state_at_30h = truth[30.0])
```

## [6. Configure the twin](@id wf-twin)

Each keyword of [`Twin`](@ref) encodes a modelling assumption
([Bayesian filtering](@ref "Bayesian filtering and data assimilation")):

| Choice | Our setting | Reasoning |
|:---|:---|:---|
| Estimator | `EnKF(N = 100, inflation = 1.02)` | 3 unknowns, mildly nonlinear: the EnKF is accurate and robust |
| `estimate` | `[:r]` | ``c`` is already a state; ``r`` is the unknown parameter |
| `prior` over ``[h, c, r]`` | means ``[1.6, 0.45, 0]``, sd ``[0.2, 0.05, 0.005]`` | design values, wide enough to contain the truth; no idea whether it is clogging |
| `process_noise` ``q`` | to be tuned in step 7 | model error on ``h``, unmodelled changes of ``c``, slow drift of ``r`` |
| `bounds` | ``h \ge 0``, ``c \ge 0.05`` | physical limits |

```@example wf
prior = MvNormal([1.6, 0.45, 0.0], Diagonal([0.2, 0.05, 0.005] .^ 2))

build(q) = Twin(model, sensors; name = "tank-7",
    estimator = EnKF(N = 100, inflation = 1.02), estimate = [:r], prior,
    process_noise = q, bounds = ([0.0, 0.05, -Inf], fill(Inf, 3)), rng = Xoshiro(1))
nothing # hide
```

## [7. Evaluate and tune](@id wf-tune)

We run the twin through the synthetic data and measure, after a 5-hour spin-up:

- **accuracy**: the RMSE of ``h`` and of the unobserved ``c`` against the truth;
- **coverage**: how often the true ``c`` lies within ±2σ of the estimate (ideal ≈ 0.95);
- **mean NIS**: the normalized innovation squared, ``d^2/(s_h^2 + \sigma^2)``, computed
  *before* each update. The ideal is ≈ 1; ≫ 1 means over-confident, ≪ 1 under-confident
  ([Innovations](@ref "Without ground truth (real data): innovations")).

The NIS needs no ground truth, so it is also the statistic to monitor on real data.

```@example wf
function evaluate(twin)
    err_h, err_c, inside, nis = Float64[], Float64[], Bool[], Float64[]
    for obs in data
        predict!(twin, obs.t)
        d = obs.y[1] - twin[:h]                                  # innovation
        push!(nis, d^2 / (uncertainty(twin, :h)^2 + σ^2))
        assimilate!(twin, obs)
        h_true, c_true = truth[obs.t]
        push!(err_h, twin[:h] - h_true)
        push!(err_c, twin[:c] - c_true)
        push!(inside, abs(twin[:c] - c_true) < 2uncertainty(twin, :c))
    end
    k = 21:length(data)                                          # skip 5 h spin-up
    return (rmse_h = sqrt(mean(err_h[k] .^ 2)), rmse_c = sqrt(mean(err_c[k] .^ 2)),
            coverage_c = mean(inside[k]), mean_NIS = mean(nis[k]),
            r = twin[:r], r_sd = uncertainty(twin, :r))
end

open_loop = Twin(model, sensors; estimator = NoAssimilation(N = 100), estimate = [:r],
    prior, rng = Xoshiro(1))
settings = ["open loop (no data)" => open_loop,
            "q = 0 (perfect model)" => build([0.0, 0.0, 0.0]),
            "q small"  => build([0.005, 0.001, 0.0001]),
            "q medium" => build([0.005, 0.005, 0.0002]),
            "q large"  => build([0.05, 0.03, 0.005])]
results = DataFrame([(; setting = name, evaluate(tw)...) for (name, tw) in settings])
```

How to read this table (the true clogging rate is ``r = 0.003``):

- **Open loop**: without data, the error in ``h`` is an order of magnitude larger.
  Assimilation is clearly worth it.
- **``q = 0``** assumes the model is perfect. After the debris event, which the model
  cannot represent, the twin becomes *confidently wrong*. Coverage of ``c`` drops, and
  the clogging rate is biased, with a very small reported uncertainty. This is filter
  divergence in the making.
- **Medium ``q``** gives good accuracy, coverage near 0.95 and a NIS near 1: accurate
  *and* honest.
- **Large ``q``** still tracks the level, but the estimates are vague (a large `r_sd`),
  and NIS ≪ 1 shows the twin is under-confident.

We choose the medium setting. As a rule, increase ``q`` until coverage and NIS are
consistent, and stop before accuracy degrades.

!!! tip "Check before moving on"
    RMSE well below open loop; coverage of the key unknowns ≈ 0.9–0.97; mean NIS
    ≈ 1 (0.5–2 is usually fine). If NIS ≫ 1, increase `process_noise` or `inflation`.
    If parameters never move, revisit step 4.

## [8. Replay as a live stream](@id wf-replay)

Next, run the same data through the live machinery ([Running live](@ref)): a
[`TableSource`](@ref) replays a table as if it arrived in real time, and **sinks** act
on every update. Here an alert sink forecasts the next 24 h every hour, and raises an
alarm when overflow becomes more likely than not.

```@example wf
q = [0.005, 0.005, 0.0002]
twin = build(q)
table = (t = [o.t for o in data], h = [o.y[1] for o in data])

alerts = DataFrame(t = Float64[], P_overflow_24h = Float64[])
overflow_alarm = CallbackSink() do tw, obs
    (obs === nothing || tw.t % 1 != 0) && return               # hourly
    fc = forecast(tw, tw.t + 24; saveat = tw.t:0.5:(tw.t + 24), nsamples = 100)
    p = mean(maximum(fc[:h]; dims = 1) .> 3.0)
    p > 0.5 && push!(alerts, (tw.t, p))
end
snapshots = SnapshotSink(joinpath(mktempdir(), "tank-7.jld2"); every = 20)

runner = run!(twin, TableSource(table); sinks = [overflow_alarm, snapshots])
```

```@example wf
first(alerts, 3)
```

```@example wf
(c = twin[:c], c_true = truth[30.0][2], r = twin[:r], r_sd = uncertainty(twin, :r))
```

The twin raises its first alarm many hours before the overflow would happen, while the
level is still well below 3 m. The alarm is driven by the *learned* clogging rate
extrapolated forward, not by the current level.

## [9. Support the decision](@id wf-decide)

Now answer the operators' question with [`scenario`](@ref)s. Each option is an
intervention ([What-if scenarios](@ref "What-if scenarios as interventions")), evaluated
on the same samples of the current belief:

- *do nothing*;
- *clean the outlet* in 2 h, or in 10 h: reset ``c`` to 0.5 ([`SetState`](@ref));
- *reduce production* by 15% now: ``s = 0.85`` ([`SetParameter`](@ref)).

```@example wf
t0 = twin.t
saveat = t0:0.25:(t0 + 24)
options = [
    forecast(twin, t0 + 24; saveat, name = "do nothing"),
    scenario(twin, SetState(t0 + 2, :c => 0.5), t0 + 24; saveat, name = "clean outlet in 2 h"),
    scenario(twin, SetState(t0 + 10, :c => 0.5), t0 + 24; saveat, name = "clean outlet in 10 h"),
    scenario(twin, SetParameter(t0, :s => 0.85), t0 + 24; saveat, name = "reduce inflow 15%"),
]

function summarize(fc)
    H = fc[:h]
    over = vec(any(H .> 3.0; dims = 1))
    first_over = [fc.t[findfirst(>(3.0), H[:, k])] for k in axes(H, 2) if over[k]]
    (option = fc.name, P_overflow = mean(over),
     median_overflow_time = isempty(first_over) ? missing : median(first_over),
     peak_level_p95 = quantile(vec(maximum(H; dims = 1)), 0.95))
end
DataFrame(summarize.(options))
```

The answer is a **probability**, with timing. Doing nothing overflows with high
probability in about 20 h. Reducing production only delays the overflow, with a risk
that remains. Cleaning the outlet, even 10 h from now, removes the risk. Operators can
therefore schedule the cleaning in the next shift rather than as an emergency.

!!! warning "A forecast is only as good as its assumptions"
    These forecasts assume the clogging continues at the learned rate, and they do not
    include future debris events. Report them with their assumptions, and re-run them
    as new data arrives: the twin updates continuously.

## [10. Deploy and maintain](@id wf-deploy)

Going live means replacing the synthetic data with the real feed. The rest of the code
stays the same:

1. **Calibrate** fixed parameters once on historical or commissioning data
   ([Calibration and persistence](@ref)).
2. **Connect** the plant data to a [`ChannelSource`](@ref) or a custom source, with
   timestamps converted to model time ([Running live](@ref)).
3. **Run** with `on_error = :skip`, a [`WallClock`](@ref) for sensor outages, and a
   [`SnapshotSink`](@ref) so that a restart resumes where it left off
   ([`restore!`](@ref)).
4. **Monitor the innovations.** The mean NIS should stay near 1, and the innovations
   should not stay on one side of zero. A persistent change means the plant no longer
   behaves like the model: a new fault, a replaced component, a drifting sensor. That
   is itself valuable information. Investigate, update the model, and repeat steps 4–7.
5. **Keep the synthetic test** (steps 5–7) as an automated regression test, so that
   model changes never silently degrade the twin.

## Summary

| Step | Code | Done when |
|:---|:---|:---|
| 1 | — | the question, horizon and actions are written down |
| 2 | `ODEModel(f, u0, p; states, params)` | simulations are physically plausible |
| 3 | `ObservationModel(model, vars, σ)` | every sensor maps to the model, with a realistic σ |
| 4 | `Vardr.propagate` / `forecast` with perturbed values | unknowns change outputs by ≫ σ |
| 5 | a "real" model with unmodelled effects | noisy synthetic data exists |
| 6 | `Twin(...; estimator, estimate, prior, process_noise, bounds)` | every choice is justified |
| 7 | `predict!` / `assimilate!` loop with RMSE, coverage, NIS | accurate and honest |
| 8 | `run!(twin, TableSource(...); sinks)` | alerts fire sensibly on replayed data |
| 9 | `scenario`, `SetState`, `SetParameter`, `compare` | options ranked by risk |
| 10 | `ChannelSource`, `WallClock`, `SnapshotSink`, `restore!` | monitored in production |
