```@meta
CurrentModule = Vardr
```

# Prediction, calibration and verification

## Forecasting: the predictive distribution

A forecast answers "what will the system do over the next ``\tau`` time units, given
everything observed so far?" Formally, it is the **predictive distribution**

```math
p(u_{t+\tau} \mid y_{1:k}) = \int p(u_{t+\tau} \mid z_k)\, p(z_k \mid y_{1:k})\, \mathrm{d}z_k .
```

Vardr evaluates it by Monte Carlo: draw samples ``z^{(s)}`` from the current belief and
integrate each one forward with its own parameters:

```math
u_{t+\tau}^{(s)} = \mathcal M_{t \to t+\tau}\big(u^{(s)}, \theta^{(s)}\big), \qquad s = 1, \dots, S.
```

The forecast spread therefore combines **state** uncertainty and **parameter**
uncertainty, and it grows with the horizon. Chaotic systems grow it exponentially, at a
rate set by the largest Lyapunov exponent [Lorenz1963](@cite). The horizon over which a
twin forecast is useful is therefore limited, and the ensemble spread is an estimate of
that limit (see `examples/lorenz63_twin.jl`).

Vardr's forecasts are **deterministic given the sample**: model error ``\eta`` is not
added. They represent uncertainty about the present system, not about future random
disturbances. If future disturbances matter, for example weather, include them as
explicit scenarios or as uncertain inputs.

!!! note "In Vardr"
    [`forecast`](@ref)`(twin, t_end; nsamples = S, saveat)`. For an EnKF with
    `S = N`, each member is used exactly once; otherwise samples are drawn according to
    the belief's weights. Any probability is a sample average, for example
    ``P(\max_t x > c) \approx \frac1S \sum_s \mathbb 1[\max_t x^{(s)}(t) > c]``, which
    is `mean(maximum(fc[:x]; dims = 1) .> c)` in code.

## What-if scenarios as interventions

A what-if question asks for the predictive distribution **under an action** ``a``: a
change of parameter, state or input at a time ``t_a``. Vardr *replaces* the affected
quantity in the model at ``t_a`` and integrates on, which is an *intervention* on the
mechanistic model:

```math
p(u_{t+\tau} \mid y_{1:k}, \mathrm{do}(a)).
```

Because each option is evaluated on the **same samples** of the current belief,
differences between scenarios reflect the actions, not sampling noise. This is the
method of *common random numbers*. Decisions should then be based on decision-relevant
summaries: the expected value of a cost, the probability of violating a limit, or a
quantile. Do not use the mean trajectory alone: the mean of a nonlinear function is not
the function of the mean.

!!! note "In Vardr"
    [`scenario`](@ref)`(twin, interventions, t_end)` with [`SetParameter`](@ref),
    [`SetState`](@ref) and [`Event`](@ref). Internally these are SciML
    `DiscreteCallback`s at `tstops`, so the solver stops exactly at ``t_a``.

## Calibration: maximum likelihood

Online estimation learns parameters sequentially. **Calibration** fits them to a batch
of historical data at once. Under the observation model, and treating the model as
exact over the calibration window, the likelihood of the data is

```math
\mathcal L(\theta) = \prod_k \mathcal N\big(y_k;\ h(u_k(\theta), \theta),\ R\big),
\qquad u_k(\theta) = \mathcal M_{t_0 \to t_k}(u_0, \theta),
```

and the maximum-likelihood estimate minimizes the negative log-likelihood (dropping
constants):

```math
\hat\theta = \arg\min_\theta\ \frac12 \sum_k r_k(\theta)^\top R^{-1} r_k(\theta),
\qquad r_k(\theta) = y_k - h\big(u_k(\theta), \theta\big).
```

With diagonal ``R`` this is weighted least squares, each residual being scaled by its
sensor's noise. Missing values are left out of ``r_k``.

Calibration and filtering complement each other:

| | Calibration (`calibrate!`) | Filtering (`estimate = [...]`) |
|:---|:---|:---|
| Uses | all data at once, a deterministic model | data sequentially, model error allowed |
| Gives | a point estimate ``\hat\theta`` | a distribution that evolves over time |
| Assumes | parameters are constant in the window | parameters may drift |
| Typical use | commissioning, from historical data | operation |

For a full Bayesian treatment of calibration, including model discrepancy, see
[KennedyOHagan2001](@cite).

!!! note "In Vardr"
    [`calibrate!`](@ref)`(twin, data, [:k, :c]; alg)` minimizes exactly the objective
    above with any Optimization.jl optimizer. It then writes ``\hat\theta`` into the
    twin's fixed parameters, or recentres the members of parameters that are also
    estimated online.

## Identifiability

A parameter is **identifiable** if the data can determine it. A parameter is
*structurally* non-identifiable if different values produce identical outputs: for
example, only a product ``ab`` affects the measurement. It is *practically*
non-identifiable if the output differences are smaller than the noise, or the data does
not excite the relevant dynamics [RaueEtAl2009](@cite).

A quick practical check before building a twin: simulate the outputs with each
parameter perturbed by its prior uncertainty, and compare the change in ``h`` with the
sensor noise ``\sigma``. Changes much smaller than ``\sigma``, over the time scales of
interest, mean the data will not inform that parameter. Fix the parameter, measure
something else, or accept that its uncertainty will stay at the prior. The
[Step-by-step workflow](@ref) shows this check in code.

## Verifying a twin: is the uncertainty honest?

A twin is useful for decisions only if its uncertainty is **calibrated**: when it says
"90% interval", the truth should fall inside about 90% of the time. Over-confidence is
dangerous; under-confidence wastes information. The tools below come from ensemble
forecast verification [Hamill2001, GneitingRaftery2007](@cite) and from data-assimilation
diagnostics [DesroziersEtAl2005](@cite).

### With ground truth (synthetic experiments)

Before using real data, run an **observing-system simulation experiment** (OSSE, also
called a *twin experiment*). Simulate a "true" system with known parameters, generate
noisy observations from it, and check that the twin recovers the truth. Every Vardr
example is such an experiment. Check:

- **Accuracy**: the root-mean-square error of the posterior mean is well below that of
  an open-loop simulation (`NoAssimilation`).
- **Coverage**: the fraction of times the truth lies within ``\bar z \pm m\sigma``. For a
  Gaussian, this is ≈ 68% for ``m = 1`` and ≈ 95% for ``m = 2``.
- **Rank histogram**: the rank of the truth among the sorted members should be uniform
  [Hamill2001](@cite). A U shape means too little spread; a dome shape means too much.

### Without ground truth (real data): innovations

With real data the truth is unknown, but the measurement is not. The **innovation** is
the difference between a measurement and the twin's prediction of it, *before* the
update:

```math
d_k = y_k - \overline{h(z_k^f)}, \qquad
S_k = \operatorname{Cov}\big(h(z_k^f)\big) + R .
```

If the twin is consistent, ``d_k`` has mean zero and covariance ``S_k``
[DesroziersEtAl2005](@cite). Two practical statistics:

- **Bias**: the mean of ``d_k`` over time should be ≈ 0. A persistent bias indicates a
  model error that the twin cannot explain: missing physics, a drifting sensor, or a
  parameter outside the prior.
- **Normalized innovation squared**: ``\mathrm{NIS}_k = d_k^\top S_k^{-1} d_k`` should
  average to the number of observed outputs (1 for a single sensor). If it is much
  larger, the twin is over-confident: increase process noise or inflation. If it is
  much smaller, the twin is under-confident, or ``R`` is too large.

Monitoring innovations online is also a powerful **anomaly detector**. A sudden run of
large innovations means the real system has stopped behaving like its twin.

!!! note "In Vardr"
    Every prediction (`kind = :predict`) and update (`kind = :update`) is recorded in
    [`history`](@ref). For sensors that observe states directly, the predicted mean and
    standard deviation of the observed state are in the `:predict` rows, which is
    enough to compute ``d_k`` and the NIS. The [Step-by-step workflow](@ref) does this.

## Sequential processing of live data

Filtering is sequential: observations must be assimilated in time order. In live
operation, data can arrive late, out of order, or not at all. Vardr's runner uses these
rules:

- An observation newer than the twin is assimilated normally.
- During silence, an optional wall clock advances the twin by prediction only, so the
  uncertainty grows honestly. These advances are **provisional**. If a delayed
  observation then arrives, the twin rolls back to the last checkpoint and assimilates
  it. This is the same as having waited for it.
- An observation older than the last *assimilated* one cannot be used without
  re-running the filter from an earlier state. It is counted as late and dropped.

!!! note "In Vardr"
    [`run!`](@ref), [`WallClock`](@ref), and the `late`, `rollbacks` and `failed`
    counters on [`TwinRunner`](@ref).
