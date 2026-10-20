```@meta
CurrentModule = Vardr
```

# Digital twins and state-space models

This chapter explains what Vardr computes and why. The next chapters cover
[Bayesian filtering](@ref "Bayesian filtering and data assimilation") (how the twin
learns from data) and
[forecasting, calibration and diagnostics](@ref "Prediction, calibration and verification").
Each section ends with a short **In Vardr** note that maps the theory onto the API.

## What is a digital twin?

The term goes back to product lifecycle management
[GrievesVickers2017](@cite). A recent consensus report by the US National Academies
defines a digital twin as

> a set of virtual information constructs that mimics the structure, context, and
> behavior of a natural, engineered, or social system (or system-of-systems), is
> dynamically updated with data from its physical twin, has a predictive capability,
> and informs decisions that realize value. The bidirectional interaction between the
> virtual and the physical is central to the digital twin.
> [NASEM2024](@cite)

Three ingredients separate a twin from an ordinary simulation:

1. **A model of one specific asset or individual**: this pump, this patient, this
   building. It is not a generic model of pumps.
2. **Continuous updating from data.** As measurements arrive, the twin's estimate of the
   asset's current state, and of its uncertain properties, is corrected. In the
   geosciences this is called *data assimilation*
   [Evensen2007, AschBocquetNodet2016](@cite).
3. **Prediction in support of decisions.** The updated twin forecasts what will happen,
   and what would happen under alternative actions, *with quantified uncertainty*
   [KapteynEtAl2021](@cite).

The loop that connects them:

```
          ┌──────────────── physical system ◀──────────── decisions ─────┐
          │                                                              │
     measurements y_k                                                    │
          ▼                                                              │
  ┌─ assimilate ─┐     ┌─ predict ─┐                ┌─ forecast / what-if ┐
  │ p(z_k|y_1:k) │ ──▶ │ p(z_k+1|  │ ──▶  ...  ──▶  │ p(z_t+τ | y_1:k,    │
  └──────────────┘     │   y_1:k)  │                │   intervention)     │
                       └───────────┘                └─────────────────────┘
```

Vardr implements the inner part of this loop: assimilation, prediction, forecasting and
what-if analysis. The decision itself is yours.

## The state-space model

Vardr describes a system in **state-space form**, the standard formulation of filtering
theory [Jazwinski1970, Sarkka2013](@cite).

**Process (dynamics).** The state ``u(t) \in \mathbb{R}^{n}`` evolves according to an
ordinary differential equation with parameters ``p``:

```math
\frac{\mathrm{d}u}{\mathrm{d}t} = f(u, p, t).
```

Between two observation times ``t_{k-1}`` and ``t_k``, we write the solution map
(the *flow*) as

```math
u_k = \mathcal{M}_{t_{k-1} \to t_k}(u_{k-1}, p).
```

The model is never perfect. Unmodelled physics, unknown inputs and discretization
errors are lumped into a **model error** ``\eta_k``:

```math
u_k = \mathcal{M}_{t_{k-1} \to t_k}(u_{k-1}, p) + \eta_k .
```

**Observation.** At time ``t_k``, sensors measure

```math
y_k = h(u_k, p, t_k) + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0, R),
```

where ``h`` is the observation operator and ``R`` the measurement-error covariance.
Usually ``h`` returns a few components of ``u`` (temperature at a sensor, the level of a
tank). It can also be any nonlinear function, such as a drug effect computed from a
concentration.

**Prior.** Before any data, our knowledge of the state is a probability distribution
``p(u_0)``.

!!! note "In Vardr"
    - ``f``, the nominal ``u_0`` and ``p``: [`TwinModel`](@ref). The flow ``\mathcal{M}``
      is computed by a SciML ODE solver [RackauckasNie2017](@cite), by default `Tsit5`
      [Tsitouras2011](@cite). ModelingToolkit models [MaEtAl2021](@cite) work too.
    - ``h`` and ``R``: [`ObservationModel`](@ref). `ObservationModel(model, [:x], σ)`
      means ``h(u) = u_x`` and ``R = \operatorname{diag}(\sigma^2)``.
    - ``y_k`` at ``t_k``: [`Observation`](@ref)`(t_k, y_k)`.
    - ``p(u_0)``: the `prior` keyword of [`Twin`](@ref).
    - ``\eta_k``: the `process_noise` keyword (see [Model error](@ref) below).

## Unknown parameters: the augmented state

Some parameters ``\theta \subset p`` are not known well enough: a heat-transfer
coefficient, a patient's drug clearance, a leak. Treating them as random variables, and
appending them to the state, turns parameter estimation into state estimation
[Kitagawa1998, Evensen2009csm](@cite):

```math
z = \begin{bmatrix} u \\ \theta \end{bmatrix},
\qquad
z_k = \begin{bmatrix} \mathcal{M}(u_{k-1}, \theta_{k-1}) \\ \theta_{k-1} \end{bmatrix} + \eta_k .
```

Parameters have no dynamics of their own: they are constant apart from the noise
``\eta_k``. Data still informs them, because the filter learns the *correlation*
between ``\theta`` and the observed outputs. If ensemble members with a larger leak
coefficient predict a lower tank level, and the measured level is low, the update shifts
the leak coefficient upwards. This is called *joint* (or *augmented*) state–parameter
estimation [MoradkhaniEtAl2005](@cite).

!!! note "In Vardr"
    `Twin(model, sensors; estimate = [:k, :c])` builds ``z = [u; k; c]``. Everything
    Vardr computes (belief, history, forecasts) is over ``z``. Parameters that are not
    in `estimate` stay fixed at their values in the model.

## Model error

Vardr models ``\eta`` as a random walk with independent components:

```math
\eta_k \sim \mathcal{N}\!\big(0,\ \Delta t_k \operatorname{diag}(q)^2\big),
\qquad \Delta t_k = t_k - t_{k-1},
```

where ``q_i`` is the standard deviation per unit time of component ``i``. The variance
grows linearly with the time between observations, as for a Wiener process, so one
setting works for irregular sampling.

Model error plays two roles:

- **On states**, it represents real unpredictability: disturbances and missing physics.
  Without it, the twin trusts its model too much and stops following the data, which is
  called *filter divergence* [Jazwinski1970](@cite).
- **On parameters**, it is an *artificial evolution* that lets the parameter ensemble
  keep a nonzero spread, and lets estimates follow slow real changes such as wear or
  fouling [Kitagawa1998, LiuWest2001](@cite).

!!! note "In Vardr"
    `process_noise = q` (a scalar, or one value per component of ``z``). It is applied
    after every [`predict!`](@ref). Members are then clipped to `bounds`, a simple way to
    enforce physical constraints such as non-negative concentrations.

## Uncertainty is represented by samples

The quantity a twin maintains is the **filtering distribution**
``p(z_k \mid y_{1:k})``: everything we know about the system now, given all data so far.
For nonlinear models this distribution has no closed form. Vardr represents it by
``N`` samples (*members*) ``z^{(1)}, \dots, z^{(N)}``, optionally with weights:

```math
p(z_k \mid y_{1:k}) \approx \sum_{i=1}^{N} w^{(i)} \, \delta\big(z - z_k^{(i)}\big).
```

Every summary is a sample statistic: the mean ``\bar z = \sum_i w^{(i)} z^{(i)}``, the
covariance, quantiles, and probabilities of events. This Monte Carlo representation is
shared by the ensemble Kalman filter (equal weights ``w^{(i)} = 1/N``) and the particle
filter (unequal weights), described in the next chapter.

!!! note "In Vardr"
    The belief is [`EnsembleBelief`](@ref) or [`ParticleBelief`](@ref). `twin[:x]` is
    the mean, [`uncertainty`](@ref) the standard deviation, and [`members`](@ref) the
    samples themselves.

## Two kinds of uncertainty

It is useful to separate:

- **Epistemic uncertainty**: lack of knowledge about the state and parameters. Data
  reduces it: the posterior spread of ``\theta`` shrinks as the twin learns.
- **Aleatoric uncertainty**: inherent randomness, such as sensor noise ``\varepsilon``
  and disturbances ``\eta``. Data does not reduce it.

A good twin reports both honestly. Overconfidence (too little spread) is the most
common failure in practice, and the most dangerous for decisions. Diagnostics that detect
it are covered in
[Verifying a twin](@ref "Verifying a twin: is the uncertainty honest?").

## Further reading

- Filtering theory: [Jazwinski1970, Sarkka2013](@cite).
- Data assimilation, mathematically: [LawStuartZygalakis2015, ReichCotter2015](@cite).
- Data assimilation in practice: [Evensen2007, AschBocquetNodet2016](@cite).
- Digital twins: [NASEM2024, KapteynEtAl2021](@cite).
