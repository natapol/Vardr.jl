```@meta
CurrentModule = Vardr
```

# Case study: a bioprocess twin

This chapter builds a twin of a **fed-batch bioreactor** that makes a recombinant protein,
the workhorse process of biopharmaceutical manufacturing. It follows the
[Step-by-step workflow](@ref) on a realistic industrial problem, explains the
bioprocess engineering behind each modelling choice, and ends with a decision that the
twin gets right and an uninformed model gets wrong.

You will see how to:

- build a model from mass balances and classical kinetics, with a feed profile driven by
  parameters,
- combine fast online sensors with slow, sparse laboratory assays (`NaN` entries),
- use a nonlinear observation function that depends on estimated parameters (the oxygen
  uptake rate),
- learn four batch-specific unknowns at once, including a hidden equipment fault,
- turn the twin's forecasts into a **risk-constrained decision**.

The same code is available as a script:
[`examples/bioreactor_fedbatch_twin.jl`](https://github.com/natapol/Vardr.jl/blob/main/examples/bioreactor_fedbatch_twin.jl).
For a twin of a *patient* rather than a process, see `examples/pkpd_patient_twin.jl`
(see [Examples](@ref)).

All code on this page runs when the documentation is built.

```@example bio
using Vardr, Distributions, LinearAlgebra, Random, Statistics, DataFrames
```

## Why twins in pharmaceutical manufacturing?

Biologics are made by living cells, and no two batches behave the same: the inoculum,
raw-material lots and equipment all vary. Quality has traditionally been ensured by
running a fixed recipe and testing the product afterwards. Regulators now encourage the
opposite approach. Under **Process Analytical Technology** (PAT), quality is built in by
measuring and controlling the process while it runs [FDA2004PAT](@cite). Under
**Quality by Design** (QbD), the manufacturer understands how process parameters affect
quality, and operates within a justified *design space* [ICHQ8R2](@cite).

Both need what a digital twin provides: estimates of quantities that cannot be measured
online (product concentration, specific rates, yields), predictions of where the batch
is heading, and the ability to test an action before taking it. Mechanistic process
models are the natural basis [NarayananEtAl2020](@cite). Combined with state estimation
they are known in bioprocessing as **soft sensors**
[LuttmannEtAl2012, Dochain2003](@cite).

## [1. The process and the decision](@id bio-purpose)

*Escherichia coli* expressing a recombinant protein is grown in a 5 L bioreactor:

1. **Batch phase** (0 to about 7 h). The medium starts with 20 g/L glucose. Cells grow
   at their maximum rate until the glucose runs out.
2. **Fed-batch phase** (7 h to harvest at 36 h). Concentrated glucose (500 g/L) is
   pumped in. The feed rate rises exponentially so that growth is held at a chosen,
   sub-maximal rate. This is the standard way to reach high cell density while avoiding
   *overflow metabolism*: when *E. coli* takes up glucose faster than it can respire
   it, the excess is excreted as acetate, which inhibits growth and protein production
   [Lee1996, YeeBlanch1992, EitemanAltman2006](@cite).

The feed cannot rise forever. Cells need oxygen to burn glucose, and their **oxygen
uptake rate** (OUR) grows with the feed. A vessel can transfer only a limited amount of
oxygen from the gas to the liquid. The transfer rate is
``\mathrm{OTR} = k_La\,(C^* - C_L)``, where ``k_La`` is the volumetric mass-transfer
coefficient and ``C^* - C_L`` the dissolved-oxygen driving force
[GarciaOchoaGomez2009](@cite). With agitation and aeration at their maximum, this
vessel can supply ``\mathrm{OTR}_{\max} = 180`` mmol O₂/L/h. If the demand exceeds it,
dissolved oxygen collapses, metabolism turns to acetate, and the batch may be lost.

So the feed must be **capped**. At t = 20 h the process engineer must choose the cap
``F_\text{cap}`` for the rest of the batch:

> Choose the largest feed cap such that the probability of the oxygen demand exceeding
> ``\mathrm{OTR}_{\max}`` before harvest is at most 5%.

A higher cap means more biomass and more product, but more oxygen risk. The right cap
depends on things nobody knows exactly for *this* batch: how fast these cells grow, how
efficiently they turn glucose into biomass, and how much glucose the pump really
delivers. This is the twin's job.

## [2. The process model](@id bio-model)

**Mass balances.** For a well-mixed reactor fed at rate ``F`` (L/h) with no outflow, the
volume grows as ``\mathrm dV/\mathrm dt = F``. For any dissolved or suspended component
with concentration ``c`` and net production rate ``r``, the amount ``cV`` obeys
``\mathrm d(cV)/\mathrm dt = rV + F c_\text{feed}``. Expanding gives a production term
plus a *dilution* term ``-(F/V)\,c``:

```math
\begin{aligned}
\frac{\mathrm dX}{\mathrm dt} &= \mu X - \frac{F}{V}X &&\text{biomass (g/L)}\\
\frac{\mathrm dS}{\mathrm dt} &= -q_S X + \frac{F}{V}(S_f - S) &&\text{glucose (g/L)}\\
\frac{\mathrm dP}{\mathrm dt} &= q_P X - \frac{F}{V}P &&\text{product (g/L)}\\
\frac{\mathrm dV}{\mathrm dt} &= F &&\text{volume (L)}
\end{aligned}
```

**Kinetics.** The specific rates use three classical laws:

- **Growth** follows **Monod** kinetics [Monod1949](@cite):
  ``\mu = \mu_{\max} S/(K_s + S)``. It is nearly ``\mu_{\max}`` when glucose is
  plentiful and proportional to ``S`` when it is scarce.
- **Glucose uptake** pays for growth and for cell **maintenance**
  [Pirt1965](@cite): ``q_S = \mu/Y_{X/S} + m_S\,S/(K_s+S)``. ``Y_{X/S}`` is the biomass
  yield (g of cells per g of glucose). The factor ``S/(K_s+S)`` switches maintenance off
  when there is no glucose left to consume.
- **Product formation** follows the **Luedeking–Piret** law
  [LuedekingPiret1959](@cite): ``q_P = \alpha\mu + q_p``. It has a growth-associated
  part and a non-growth-associated part, the *specific productivity* ``q_p``.

**Feed.** To hold growth at a set rate ``\mu_\text{set}`` under glucose limitation, the
glucose fed must equal the glucose consumed: ``F S_f = q_S X V`` with ``\mu = \mu_\text{set}``.
Because ``XV`` then grows like ``e^{\mu_\text{set} t}``, so must the feed
[Lee1996](@cite):

```math
F(t) = f_\text{pump}\,\min\!\Big(F_0\, e^{\mu_\text{set}(t - t_\text{feed})},\ F_\text{cap}\Big),
\qquad F_0 = \frac{(\mu_\text{set}/Y_{X/S} + m_S)\,X V}{S_f}\bigg|_{t_\text{feed}} .
```

``F_0`` is designed from the process development (PD) values: ``X \approx 10`` g/L,
``V = 5`` L and ``\mu_\text{set} = 0.1`` h⁻¹. The factor ``f_\text{pump}`` is the ratio
of delivered to requested flow. It should be 1, but pumps drift, and tubing wears and
creeps. In the model it is just another parameter, which the twin can estimate.

```@example bio
const PNAMES = [:mu_max, :Ks, :Yxs, :ms, :alpha, :qp, :kO, :Sf,
    :F0, :mu_set, :t_feed, :F_cap, :f_pump]

feedrate(p, t) = t < p[11] ? 0.0 : p[13] * min(p[9] * exp(p[10] * (t - p[11])), p[12])
uptake(S, p) = (p[1] / p[3] + p[4]) * S / (p[2] + S)   # specific glucose uptake q_S (g/g/h)

function bioreactor(u, p, t)
    X, S, P, V = u
    Sp = max(S, 0.0)
    mu = p[1] * Sp / (p[2] + Sp)
    F = feedrate(p, t)
    D = F / V                                          # dilution rate (1/h)
    return [mu * X - D * X,
            -uptake(Sp, p) * X + D * (p[8] - S),
            (p[5] * mu + p[6]) * X - D * P,
            F]
end

# Process development values (the "recipe"); F_cap = Inf means no cap yet
pd = [0.6, 0.05, 0.5, 0.04, 0.01, 0.004, 12.0, 500.0,
    0.024, 0.1, 7.0, Inf, 1.0]
model = ODEModel(bioreactor, [0.2, 20.0, 0.0, 5.0], pd;
    states = [:X, :S, :P, :V], params = PNAMES, tstops = [7.0])
```

`tstops = [7.0]` makes the solver stop exactly when feeding starts, because the right-hand
side jumps there (see [Models and sensors](@ref)). All feed settings are *parameters*, so a
what-if scenario can change them with [`SetParameter`](@ref).

## [3. The sensors](@id bio-sensors)

A modern bioreactor has several PAT measurements, at very different rates:

| Measurement | Measures | Every | Noise (σ) |
|:---|:---|:---|:---|
| Capacitance probe | viable biomass ``X`` | 30 min | 0.8 g/L |
| Off-gas analyser | oxygen uptake rate | 30 min | 4 mmol/L/h |
| Load cell | volume ``V`` | 30 min | 0.02 L |
| At-line glucose analyser | glucose ``S`` | 4 h | 0.1 g/L |
| Offline HPLC | product titer ``P`` | 4 h | 0.05 g/L |

- **Capacitance probes** (dielectric spectroscopy) respond to cells with intact
  membranes, so they measure viable biomass online [CarvellDowd2006](@cite).
- The **off-gas analyser** compares the O₂ fraction in the inlet and exhaust gas. With the
  gas flow, this gives the OUR. OUR is not a state: it is a function of the state and of
  the parameters. Each gram of glucose taken up consumes ``k_O`` mmol O₂, a stoichiometric
  coefficient from elemental balances (here 12 mmol/g, assumed known):
  ```math
  \mathrm{OUR} = k_O\, q_S(S;\ \mu_{\max}, Y_{X/S}, m_S)\, X .
  ```
  This is a **nonlinear observation operator that depends on estimated parameters**. The
  twin evaluates it with each ensemble member's own ``\mu_{\max}`` and ``Y_{X/S}``.
- **Glucose and titer** come from samples taken by hand and analysed in the lab. Between
  samples, their entries are `NaN`, and the twin uses only what was measured at each time
  (see [Models and sensors](@ref)).

```@example bio
our(u, p) = p[7] * uptake(max(u[2], 0.0), p) * u[1]     # mmol O2 / L / h

σ = [0.8, 4.0, 0.02, 0.1, 0.05]
sensors = ObservationModel((u, p, t) -> [u[1], our(u, p), u[4], u[2], u[3]], σ;
    names = [:X_cap, :OUR, :V_load, :S_atline, :P_hplc]);
```

## [4. What can the data tell us?](@id bio-ident)

Before choosing what to estimate, check what the sensors can see (see
[Identifiability](@ref)). We change one parameter at a time and record the largest
change in each measurement over the first 20 h, in units of that sensor's noise σ. Fewer
than about 1σ is invisible. Many σ is easily learned.

```@example bio
function simulate(p; ts = 0.5:0.5:20.0)
    u, t, out = [0.2, 20.0, 0.0, 5.0], 0.0, Vector{Float64}[]
    for tk in ts
        u = Vardr.propagate(model, u, p, t, tk)
        t = tk
        push!(out, sensors.h(u, p, tk))
    end
    return reduce(hcat, out)
end
base = simulate(pd)
function visibility(name, factor)
    p = copy(pd)
    p[findfirst(==(name), PNAMES)] *= factor
    Δ = maximum(abs.(simulate(p) .- base); dims = 2) ./ σ
    return round.(vec(Δ); digits = 1)
end
changes = [(:mu_max, 0.9), (:Yxs, 0.9), (:qp, 0.9), (:f_pump, 0.95), (:ms, 2.0), (:Ks, 2.0)]
DataFrame([(; change = "$(name) × $(f)", (sensors.names .=> visibility(name, f))...)
           for (name, f) in changes])
```

How to read the table:

- **``\mu_{\max}``, ``Y_{X/S}``** strongly affect OUR and glucose, so the data will pin
  them down.
- **``q_p``** is seen *only* by the HPLC, so it is learned only when a titer result
  arrives.
- **``f_\text{pump}``**: a 5% error changes each 30-min reading by less than 2σ. But it
  shows up as a *trend* in the volume, so it accumulates over many readings, and only
  once the feed is large.
- **``K_s``** is almost invisible: glucose is either far above ``K_s`` (batch phase) or
  near zero (fed batch). It is fixed at its literature value.
- **``m_S``** would be identifiable. It is fixed because it is well characterized for this
  strain, and estimating it alongside ``Y_{X/S}`` would make the two compete for the same
  OUR signal.

So the twin estimates ``\theta = (\mu_{\max}, Y_{X/S}, q_p, f_\text{pump})``.

## [5. A synthetic batch](@id bio-osse)

To test the twin we need a "real" batch with a known answer (see
[Verifying a twin](@ref "Verifying a twin: is the uncertainty honest?")). This batch
differs from the PD values in every estimated parameter. The cells grow 13% slower,
yield 12% less biomass and produce 60% more protein. The pump delivers 8% less than
requested, with ±3% fluctuations every 30 min. The inoculum is also 25% larger than
assumed.

```@example bio
batch = copy(pd)
batch[[1, 3, 6, 13]] = [0.52, 0.44, 0.0065, 0.92]   # mu_max, Yxs, qp, f_pump

function run_plant(p, u, t0, t1; rng, dt = 0.5)
    traj = Dict{Float64, Vector{Float64}}()
    t = t0
    for tk in (t0 + dt):dt:t1
        pk = copy(p)
        pk[13] *= 1 + 0.03 * randn(rng)                # pump fluctuations
        u = Vardr.propagate(model, u, pk, t, tk)
        t = tk
        traj[tk] = u
    end
    return traj
end
rng = Xoshiro(3)
truth = run_plant(batch, [0.25, 20.0, 0.0, 5.0], 0.0, 20.0; rng)

data = map(0.5:0.5:20.0) do t
    u = truth[t]
    sampled = t % 4 == 0                             # lab samples every 4 h
    Observation(t, [u[1] + σ[1] * randn(rng), our(u, batch) + σ[2] * randn(rng),
        u[4] + σ[3] * randn(rng),
        sampled ? u[2] + σ[4] * randn(rng) : NaN,
        sampled ? u[3] + σ[5] * randn(rng) : NaN])
end
data[end]
```

## [6. Configure the twin](@id bio-twin)

Each setting has a reason:

- **Estimator**: `EnKF(N = 200)`. There are 8 unknowns (4 states and 4 parameters), the
  model is mildly nonlinear, and observations are frequent. A small inflation (1.02)
  offsets sampling error (see [Bayesian filtering](@ref "Bayesian filtering and data assimilation")).
- **Prior**: the PD values, with the batch-to-batch variation seen in development (about
  15% on ``\mu_{\max}``, 10% on ``Y_{X/S}``, 50% on ``q_p``, 5% on the pump).
- **Model error**: small on states, to absorb the pump fluctuations and the simple
  kinetics. Very small on parameters, enough to keep their spread from collapsing (see
  [Joint state–parameter estimation in practice](@ref)).
- **Bounds**: concentrations, volume and parameters are non-negative.

```@example bio
estimate = [:mu_max, :Yxs, :qp, :f_pump]
prior = MvNormal([0.2, 20.0, 0.0, 5.0, 0.6, 0.5, 0.004, 1.0],
    Diagonal([0.05, 0.5, 0.001, 0.05, 0.1, 0.05, 0.002, 0.05] .^ 2))
make_twin(estimator) = Twin(model, sensors; name = "batch-2417", estimator, estimate, prior,
    process_noise = [0.05, 0.005, 0.005, 0.002, 0.002, 0.002, 0.0001, 0.002],
    bounds = (zeros(8), fill(Inf, 8)), rng = Xoshiro(1))
twin = make_twin(EnKF(N = 200, inflation = 1.02))
```

## [7. Follow the batch](@id bio-track)

```@example bio
fmt(twin, s) = string(round(twin[s]; sigdigits = 3), " ± ", round(uncertainty(twin, s); sigdigits = 2))
rows = []
for obs in data
    step!(twin, obs)
    obs.t in (4.0, 7.0, 12.0, 16.0, 20.0) || continue
    push!(rows, (; t = obs.t, (s => fmt(twin, s) for s in estimate)...,
        X = round(twin[:X]; digits = 2), X_true = round(truth[obs.t][1]; digits = 2)))
end
DataFrame(identity.(rows))
```

The true values are ``\mu_{\max} = 0.52``, ``Y_{X/S} = 0.44``, ``q_p = 0.0065`` and
``f_\text{pump} = 0.92``. The order in which the twin learns follows the identifiability analysis:

- **During the batch phase (0 to 7 h)**, glucose is plentiful and cells grow at
  ``\mu_{\max}``. The rising OUR and biomass reveal ``\mu_{\max}`` and ``Y_{X/S}``.
- **At 4 h and 8 h** the first titers are still tiny, so ``q_p`` stays at its prior.
  **By 12 h** two meaningful titers have arrived, and ``q_p`` is pinned down.
- **The pump error** cannot be seen until enough feed has been delivered for the volume
  trend to differ from the setpoint. At 12 h only about 0.15 L has been fed, and an 8%
  shortfall is about 0.01 L, below the load-cell noise. **By 20 h** the twin has found that
  the pump delivers 93 ± 2% of its setpoint.

This is joint state–parameter estimation at work (see
[Unknown parameters: the augmented state](@ref)). No sensor measures ``f_\text{pump}``
or ``q_p``, but the ensemble learns how they correlate with what is measured.

## [8. Choose the feed cap](@id bio-decide)

**A chance constraint.** The decision is an optimization under uncertainty:

```math
\max_{F_\text{cap}}\ \mathbb E\big[P(t_h)V(t_h)\big]
\quad\text{subject to}\quad
\Pr\Big(\max_{t \le t_h}\mathrm{OUR}(t) > \mathrm{OTR}_{\max}\Big) \le 0.05 .
```

Requiring a constraint to hold with a given probability, rather than always, is called
**chance-constrained** optimization [CharnesCooper1959](@cite). The twin's forecast
estimates both quantities directly. For each candidate cap, run a
[`scenario`](@ref) to harvest and evaluate the constraint on every sample (see
[What-if scenarios as interventions](@ref)).

A detail matters here. OUR depends on the estimated parameters, so each forecast sample
must be paired with *its own* parameters. With `nsamples` equal to the ensemble size
(the default), sample ``k`` of an EnKF forecast starts from member ``k``. So
[`members`](@ref)`(twin)[:, k]` gives sample ``k``'s parameters.

```@example bio
const OTR_MAX, T_DECIDE, T_HARVEST = 180.0, 20.0, 36.0

function assess(tw, F_cap)
    fc = scenario(tw, SetParameter(T_DECIDE, :F_cap => F_cap), T_HARVEST;
        saveat = T_DECIDE:0.25:T_HARVEST)
    Z = members(tw)
    X, S, P, V = fc[:X], fc[:S], fc[:P], fc[:V]
    peak = map(axes(X, 2)) do k
        p = copy(pd)
        p[[1, 3, 6, 13]] = Z[5:8, k]                   # this sample's parameters
        maximum(our([X[j, k], S[j, k]], p) for j in axes(X, 1))
    end
    product = P[end, :] .* V[end, :]                   # grams of protein at harvest
    return (risk = mean(peak .> OTR_MAX), product = mean(product),
        lo = quantile(product, 0.05), hi = quantile(product, 0.95))
end

caps = [Inf, 0.25, 0.225, 0.2125, 0.20, 0.1875, 0.175, 0.15]
assessed = [assess(twin, c) for c in caps]
DataFrame(F_cap = [c == Inf ? "none" : string(c) for c in caps],
    P_overload = [a.risk for a in assessed],
    product_g = [round(a.product; digits = 1) for a in assessed],
    product_90pct = ["$(round(a.lo; digits = 1))–$(round(a.hi; digits = 1))" for a in assessed])
```

```@example bio
best = maximum(c for (c, a) in zip(caps, assessed) if a.risk <= 0.05)
```

Uncapped, the exponential feed certainly overloads the vessel. Each step down trades a
little product for a lot of safety. The twin recommends the largest cap that meets the
5% constraint.

**What if we had no twin?** The same analysis with an open-loop model is shown below.
It starts from the PD prior and uses the recipe, but assimilates no data.

```@example bio
open_loop = make_twin(NoAssimilation(N = 200))
predict!(open_loop, T_DECIDE)
ol = [assess(open_loop, c) for c in caps]
DataFrame(F_cap = [c == Inf ? "none" : string(c) for c in caps],
    P_overload = [a.risk for a in ol], product_g = [round(a.product; digits = 1) for a in ol])
```

The open-loop model cannot support the decision:

- **It cannot meet the constraint at all.** Without data it cannot even tell whether
  the vessel is overloaded *now*. In about 30% of its samples, a little more glucose is
  left than in reality, and the oxygen demand at 20 h is already above
  ``\mathrm{OTR}_{\max}``. No feed cap can change that. A single off-gas reading rules
  it out, which is what the twin does.
- **It underestimates the product by more than 20%**, because it uses the PD productivity.

## [9. Check against the plant](@id bio-check)

In a synthetic study we can run the real batch under each choice, with the same pump
fluctuations:

```@example bio
function plant_outcome(F_cap)
    p = copy(batch)
    p[12] = F_cap
    tr = run_plant(p, truth[T_DECIDE], T_DECIDE, T_HARVEST; rng = Xoshiro(99))
    return (peak_OUR = round(maximum(our(u, p) for u in values(tr)); digits = 1),
        product_g = round(tr[T_HARVEST][3] * tr[T_HARVEST][4]; digits = 1))
end
DataFrame([(F_cap = c == Inf ? "none" : string(c), plant_outcome(c)...,
            twin_forecast_g = round(a.product; digits = 1)) for (c, a) in zip(caps, assessed)])
```

Under the recommended cap, the real batch stays below ``\mathrm{OTR}_{\max} = 180``. Its
product falls inside the twin's 90% interval, close to the forecast mean. A cap one step
higher would have overloaded the vessel, as the twin predicted.

The twin also knows that the pump under-delivers. A recipe-based calculation that
assumes ``f_\text{pump} = 1`` would choose a more cautious cap and give up product for
no reason. If the pump had *over*-delivered, the recipe would have overloaded the
vessel.

## [10. In production](@id bio-deploy)

To run this twin on a real line:

- **Data in.** Feed the online signals (capacitance, off-gas, load cell) through a
  [`ChannelSource`](@ref) from the plant historian or OPC UA server, with a
  [`WallClock`](@ref) so the twin keeps predicting through gaps (see
  [Running live](@ref)). Each lab result arrives as an `Observation` with `NaN`
  for the online channels.
- **Late lab results.** HPLC results come back hours after sampling. The run loop
  assimilates observations in time order, and counts an observation that arrives after
  the twin has moved past it as late. To use the result, keep a [`snapshot`](@ref) at
  each sampling time, then [`restore!`](@ref) it and replay the buffered online data,
  including the lab value.
- **Alarms.** Add a sink that forecasts the peak OUR every hour and raises an alarm
  when the overload probability exceeds a threshold. This is a predictive version of the
  usual dissolved-oxygen alarm.
- **Monitor the twin itself.** Track the innovations of each sensor (see
  [Without ground truth (real data): innovations](@ref)). A sudden bias in one channel
  means either a sensor fault (such as a fouled capacitance probe) or a process change
  the model does not describe (such as the onset of acetate formation).
- **Learn across batches.** Use each completed batch with [`calibrate!`](@ref) to update
  the PD values and the prior for the next batch.
- **Validation.** In a GMP setting a model used for decisions is part of the control
  strategy. Its intended use, validation evidence (synthetic studies like this one,
  plus historical batches) and lifecycle management must be documented [ICHQ8R2](@cite).

## Summary

| Step | In this case |
|:---|:---|
| Purpose | Feed cap at 20 h: maximize protein with ≤ 5% risk of oxygen overload |
| Model | Fed-batch mass balances, Monod growth, Pirt maintenance, Luedeking–Piret product, exponential feed with a cap |
| Sensors | Capacitance, off-gas OUR (nonlinear, parameter-dependent), load cell every 30 min; glucose and HPLC every 4 h |
| Identifiability | Estimate ``\mu_{\max}``, ``Y_{X/S}``, ``q_p``, ``f_\text{pump}``; fix ``K_s`` (invisible) and ``m_S`` (known, confounded) |
| Estimator | `EnKF(N = 200, inflation = 1.02)` with small parameter noise |
| Result | All four unknowns learned by 20 h, including the hidden pump error |
| Decision | Chance-constrained scenario scan; the recommended cap keeps the real batch below the limit |
| Value | The open-loop model is too uncertain to decide and underestimates product by more than 20% |

**Further reading.**

- Bioprocess modelling and estimation: [Dochain2003, NarayananEtAl2020](@cite).
- Soft sensors: [LuttmannEtAl2012](@cite).
- High-cell-density *E. coli* culture: [Lee1996](@cite).
- Oxygen transfer: [GarciaOchoaGomez2009](@cite).
- The regulatory framework: [FDA2004PAT, ICHQ8R2](@cite).
