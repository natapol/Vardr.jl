```@meta
CurrentModule = Vardr
```

# State and parameter estimation

A twin's **belief** is a cloud of `N` members: possible states of the real system. Each
time data arrives, the twin:

1. **predicts**: integrates every member up to the observation time and adds process
   noise ([`predict!`](@ref));
2. **updates**: moves or reweights the members towards what the sensors saw
   ([`assimilate!`](@ref)).

[`step!`](@ref)`(twin, obs)` does both. The posterior mean is the twin's best guess, and
the spread is its uncertainty.

!!! info "Theory"
    The equations behind every estimator on this page, with references, are in
    [Bayesian filtering and data assimilation](@ref). The meaning of `prior`,
    `estimate` and `process_noise` is explained in
    [Digital twins and state-space models](@ref).

```@setup est
using Vardr, Distributions, LinearAlgebra, Random, Statistics
osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]
real_system = ODEModel(osc, [1.5, 0.0], [3.0, 0.1])
truth(t) = Vardr.propagate(real_system, [1.5, 0.0], [3.0, 0.1], 0.0, t)
rng = Xoshiro(42)
measurements = [Observation(t, truth(t)[1] + 0.05 * randn(rng)) for t in 0.1:0.1:10.0]
model = ODEModel(osc, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
sensors = ObservationModel(model, [:x], 0.05)
```

## Building the twin: every keyword explained

```julia
Twin(model, sensors;
     estimator     = EnKF(N = 100),
     estimate      = [:k],
     prior         = MvNormal(μ, Σ),
     process_noise = [0.01, 0.01, 0.001],
     bounds        = (lower, upper),
     t0            = 0.0,
     rng           = Xoshiro(1),
     name          = "pump-7")
```

### `estimate`: joint state and parameter estimation

Parameters listed in `estimate` are appended to the state. The twin then works with the
**augmented state** `z = [u; θ]`, here `[x, v, k]`, and every member carries its own
value of `k`. Parameters that are not listed stay fixed at the model's values; read them
with [`paramvalues`](@ref).

Estimate a parameter when it is uncertain *and* the data can inform it. Estimating too
many parameters from too few sensors gives wide, correlated posteriors. Check with
[`uncertainty`](@ref) whether the twin is actually learning.

### `prior`: what you believe at the start

The prior is over the whole augmented state, in the order `[states...; estimated...]`.

| `prior` | Meaning |
|:---|:---|
| `nothing` (default) | Gaussian, mean = nominal `u0` and parameter values, 10% standard deviation |
| a `Distributions` distribution | e.g. `MvNormal(μ, Σ)`, or `product_distribution([Uniform(...), ...])` |
| a matrix | your own members, one per column (`size = (nz, N)`) |

Make the prior **wide enough to contain the truth**. An ensemble cannot learn what it
never considers, and a particle filter in particular cannot recover from a prior that
misses the truth.

### `process_noise`: how wrong the model may be

The standard deviation, per unit of time, of a random walk added to each component of
`z` after every prediction. It is a scalar, or one value per component.

- On **states**: it represents model error and unmodelled disturbances. Too little, and
  the twin becomes overconfident and stops listening to data (the filter *diverges*).
  Too much, and the estimates become noisy.
- On **estimated parameters**: a small value keeps the parameter ensemble from
  collapsing to a point, and lets the twin follow slow drift in the real system, such as
  wear, fouling or a leak that grows. Use a larger value if parameters can change
  abruptly.

It scales with `√dt`, so the same setting works for irregular sampling.

### `bounds`: physical limits

`(lower, upper)`, each a scalar or a vector over `z`. Members are clamped after every
prediction and update. Use it for quantities that cannot be negative (concentrations,
water levels, resistances):

```julia
bounds = (vcat(zeros(3), 0.0), fill(Inf, 4))
```

### `rng`, `t0`, `name`

- `rng`: pass a seeded RNG (`Xoshiro(1)`) for reproducible runs and exact
  [`snapshot`](@ref) round trips.
- `t0`: start time of the belief (default: the model's start time).
- `name`: shown when printing the twin and in runner logs.

## Choosing an estimator

| | [`EnKF`](@ref) | [`ParticleFilter`](@ref) | [`NoAssimilation`](@ref) |
|:---|:---|:---|:---|
| Update | Shifts members (linear-Gaussian assumption) | Reweights and resamples members | None (open loop) |
| Typical `N` | 20–200 | 500–10,000 | any |
| Dimension | Scales to thousands of states (with localization) | Small (≲ 10 unknowns) | any |
| Good for | Most twins; mildly nonlinear systems; many sensors | Strong nonlinearity, multimodal or non-Gaussian beliefs, bounded or discrete-like states | Simulation, baselines, synthetic data |
| Main risk | Ensemble collapse (use inflation and process noise) | Weight degeneracy (use jitter, a larger `N`, a wider prior) | n/a |

Start with the EnKF. Move to the particle filter only when the EnKF's Gaussian
assumption clearly fails, for example when the posterior is bimodal.

### EnKF tuning

```julia
EnKF(N = 100, inflation = 1.02, localization = nothing)
```

- `N`: more members give better covariance estimates. Cost grows linearly.
- `inflation`: spreads the ensemble by this factor before each update. Values of
  1.01–1.1 counteract the EnKF's tendency to become overconfident with small ensembles.
- `localization`: see [Localization](@ref guide-localization) below.

### Particle filter tuning

```julia
ParticleFilter(N = 2000, resample_threshold = 0.5, jitter = 0.05)
```

- `resample_threshold`: resample when the effective sample size drops below this
  fraction of `N`.
- `jitter`: after resampling, perturbs each particle by this fraction of the posterior
  standard deviation. This prevents all particles from collapsing onto a few copies,
  which matters most for estimated parameters (they have no dynamics of their own to
  spread them out).

```@example est
pf = Twin(model, sensors; estimator = ParticleFilter(N = 2000, jitter = 0.05),
    estimate = [:k], prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2)),
    process_noise = [0.01, 0.01, 0.005], rng = Xoshiro(3))
foreach(o -> step!(pf, o), measurements)
(k = pf[:k], k_std = uncertainty(pf, :k))
```

## [Localization](@id guide-localization)

With fewer members than unknowns (common for spatial models), the EnKF's sample
covariances contain **spurious correlations**: a sensor at one end of a pipe seems to
"inform" the other end. Localization tapers covariances to zero with distance.

```julia
EnKF(N = 20, localization = (Lxy, Lyy))
```

- `Lxy` (`nstates × noutputs`): the taper between each model state and each sensor.
- `Lyy` (`noutputs × noutputs`): the taper between sensors.

Build both from the **same** correlation function of distance, typically the
Gaspari–Cohn function. See the heated-rod example
(`examples/heat_rod_twin.jl`) for a complete version:

```julia
L(i, j) = gaspari_cohn(abs(position[i] - position[j]) / c)   # c ≈ half the cut-off distance
Lxy = [L(i, s) for i in 1:nstates, s in sensor_states]
Lyy = [L(s, r) for s in sensor_states, r in sensor_states]
```

If the sensors observe states directly (`ObservationModel(model, vars, σ)`), passing
`Lxy` alone is enough, and `Lyy` is derived from it. Estimated parameters are global, so
they always use the unlocalized update.

## Stepping manually

`step!` is `predict!` followed by `assimilate!`. Call them separately when you need to:

```@example est
twin = Twin(model, sensors; estimator = EnKF(N = 100), estimate = [:k],
    prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2)),
    process_noise = 0.01, rng = Xoshiro(1))

predict!(twin, 0.5)                 # advance without data (e.g. a missed reading)
obs = Observation(0.5, truth(0.5)[1] + 0.05 * randn(rng))
assimilate!(twin, obs)              # requires obs.t == twin.t
twin.t
```

Rules:

- Time only moves forward: `predict!(twin, t)` with `t < twin.t` throws. Data older
  than the twin cannot be assimilated (sort your data first; the live runner handles
  this for you).
- Several observations at the same time can be assimilated one after another.
- `predict!` without a later `assimilate!` is how a twin "coasts" through outages. Its
  uncertainty grows, which is exactly what you want.

## Reading the belief

| Call | Returns |
|:---|:---|
| `twin[:x]`, [`state`](@ref)`(twin, :x)` | posterior mean of a state or estimated parameter |
| [`state`](@ref)`(twin)` | mean of all model states |
| [`uncertainty`](@ref)`(twin, :x)`, `uncertainty(twin)` | posterior standard deviation |
| [`paramvalues`](@ref)`(twin)` | `name => value` for every parameter |
| [`members`](@ref)`(twin)` | the member matrix (`nz × N`), for any custom statistic |
| [`history`](@ref)`(twin)` | every prediction and update (a Tables.jl table) |
| `twin.t` | current model time |

Members let you compute anything, for example correlations between unknowns or the
probability of an event *now*:

```@example est
foreach(o -> step!(twin, o), measurements[6:end])
Z = members(twin)
(corr_x_k = cor(Z[1, :], Z[3, :]), P_k_above_3 = mean(Z[3, :] .> 3.0))
```

For a `ParticleFilter` the members are weighted. Use `mean`, `std` and `cov` of
`twin.belief`, or `Vardr.weights(twin.belief)`, rather than unweighted statistics of
`members(twin)`.

## Is the twin working? Diagnostics

Test your twin on **synthetic data** first: simulate a "real" system with known
parameters, as every example in `examples/` does. Then check:

1. **Accuracy**: the error of the posterior mean against the truth should be well below
   that of an open-loop run (`NoAssimilation`).
2. **Calibration of the uncertainty**: the truth should lie within ±2σ about 95% of the
   time. If it is much less, the twin is overconfident: increase `process_noise`,
   `inflation` or `N`. If it is always far inside, the twin is too cautious.
3. **Parameter learning**: plot `k` and `k_std` from the history. The mean should settle
   and the uncertainty shrink. If the uncertainty stays at its prior value, the data
   does not inform that parameter.

```@example est
using DataFrames
h = DataFrame(history(twin))
upd = h[h.kind .== :update, :]
err = [abs(r.x - truth(r.t)[1]) / r.x_std for r in eachrow(upd)]  # error in units of σ
(within_1σ = mean(err .< 1), within_2σ = mean(err .< 2),           # ideal ≈ 0.68 and 0.95
 final_k = upd.k[end], final_k_std = upd.k_std[end])
```

Here the twin is somewhat cautious: the truth lies within 1σ more often than the ideal
68%. That is the safe side to err on. Reducing `process_noise` on the states would
tighten the uncertainty.

For real data without ground truth, look at the **innovations**: the difference between
each measurement and the twin's prediction just before it. They should look like
zero-mean noise, with a spread that matches the predicted uncertainty plus the sensor
noise.
