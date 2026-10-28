```@meta
CurrentModule = Vardr
```

# Case study: validating Vardr against exact answers

A twin that supports decisions must be trustworthy in two separate ways:

- **Is the model right for the system?** That is the user's job, and the
  [Step-by-step workflow](@ref) and [Case study: a bioprocess twin](@ref) show how to
  test it.
- **Does the software compute what the theory says?** That is this chapter's job.

The literature on verification and validation separates these as *validation* (solving
the right equations) and *verification* (solving the equations right)
[OberkampfRoy2010](@cite).

Here, Vardr is checked on problems where the correct answer is known exactly, or can be
computed independently to high accuracy:

| Check | Vardr feature | Reference answer | Theory |
|:---|:---|:---|:---|
| [V1](@ref val-kalman) | `EnKF`, `ParticleFilter` | Kalman filter (exact for linear-Gaussian models) | [Mandel2011, CrisanDoucet2002](@cite) |
| [V2](@ref val-param) | Joint state–parameter estimation (`estimate`) | Kalman filter on the augmented state | [Kitagawa1998](@cite) |
| [V3](@ref val-nonlinear) | `ParticleFilter` on a bimodal posterior | Grid-based Bayesian filter | [Kitagawa1987](@cite) |
| [V4](@ref val-forecast) | `forecast`, `scenario`, `SetParameter` | Matrix exponential | — |
| [V5](@ref val-calibrate) | `calibrate!` | Closed-form maximum likelihood | — |
| [V6](@ref val-sbc) | The whole pipeline, statistically | Coverage of credible intervals in repeated experiments | [CookGelmanRubin2006, TaltsEtAl2018](@cite) |

Each check states its **acceptance criterion before the result**, as in a validation
protocol, and the code decides pass or fail. A summary table is at the
[end](@ref val-summary). The same checks, at smaller sizes, run in Vardr's test suite on
every change (`test/test_validation.jl`).

All code on this page runs when the documentation is built.

```@example val
using Vardr, Distributions, LinearAlgebra, Random, Statistics, DataFrames
using OptimizationOptimJL: NelderMead

verdicts = DataFrame(check = String[], criterion = String[], result = String[], pass = Bool[])
function verdict!(check, criterion, result, pass)
    push!(verdicts, (check, criterion, result, pass))
    return pass ? "PASS" : "FAIL"
end
nothing # hide
```

## The test system

A damped linear oscillator, the position of which is measured with noise:

```math
\dot x = v, \qquad \dot v = -\omega^2 x - 2\zeta\omega v,
\qquad y_k = x(t_k) + \varepsilon_k .
```

With ``\omega = 1``, ``\zeta = 0.1``, observations every ``\Delta t = 0.5``,
``\sigma = 0.1`` and process noise ``q = (0.05, 0.05)``, this is a **linear-Gaussian**
state-space model:

```math
u_k = F u_{k-1} + \eta_k,\quad F = e^{A\Delta t},\quad
\eta_k \sim \mathcal N(0, \Delta t\,\operatorname{diag}(q)^2),\qquad
y_k = H u_k + \varepsilon_k .
```

For such a model the Kalman filter is the exact Bayesian solution (see
[The Kalman filter](@ref)). The flow ``F`` is a matrix exponential, so the reference
answer needs no ODE solver. Vardr integrates the ODE numerically; tight solver
tolerances make that error negligible (``10^{-10}``).

```@example val
osc(u, p, t) = [u[2], -p[1]^2 * u[1] - 2p[2] * p[1] * u[2]]
model = TwinModel(osc, [1.0, 0.0], [1.0, 0.1]; states = [:x, :v], params = [:ω, :ζ],
    abstol = 1e-10, reltol = 1e-10)
sensors = ObservationModel(model, [:x], 0.1)

A(ω, ζ) = [0.0 1.0; -ω^2 -2ζ*ω]
Δt, K = 0.5, 40
F = exp(A(1.0, 0.1) * Δt)
q = [0.05, 0.05]
Q = Δt * Diagonal(q .^ 2)
H, R = [1.0 0.0], fill(0.1^2, 1, 1)
m0, P0 = [1.0, 0.0], Diagonal([0.5, 0.5] .^ 2)

# One realization of the system, drawn from exactly this model
data = let rng = Xoshiro(1), x = m0 .+ sqrt.(diag(P0)) .* randn(rng, 2)
    map(1:K) do k
        x = F * x .+ sqrt.(diag(Q)) .* randn(rng, 2)
        Observation(k * Δt, [x[1] + 0.1 * randn(rng)])
    end
end

# The reference: a textbook Kalman filter
function kalman(F, Q, H, R, m, P, data)
    means, sds = Vector{Float64}[], Vector{Float64}[]
    for o in data
        m, P = F * m, F * P * F' + Q                     # predict
        Kg = P * H' / (H * P * H' + R)                    # gain
        m, P = m + Kg * (o.y - H * m), (I - Kg * H) * P   # update
        push!(means, m)
        push!(sds, sqrt.(diag(P)))
    end
    return means, sds
end
kf_mean, kf_sd = kalman(F, Q, H, R, m0, Matrix(P0), data)
nothing # hide
```

## [V1. Ensemble and particle filters converge to the Kalman filter](@id val-kalman)

**Theory.** With linear dynamics and observations and Gaussian noise, both Vardr
estimators should reproduce the Kalman filter as the number of members grows:

- The stochastic EnKF with perturbed observations converges to it as ``N \to \infty``
  [Mandel2011](@cite).
- So does the particle filter, which converges to the exact posterior for any model
  [CrisanDoucet2002](@cite).

Both errors are Monte Carlo errors, so they should shrink like ``N^{-1/2}``.

**Acceptance criteria.**

1. Errors decrease as ``N^{-1/2}``: the fitted log–log slope is between −0.65 and −0.35.
2. At ``N = 3200``, the EnKF mean is within 0.05 posterior standard deviations of the
   Kalman mean.
3. At ``N = 3200``, the EnKF standard deviation is within 3% of the Kalman value.

We measure the error over all 40 assimilation cycles, averaged over three random seeds.
The mean error is ``|\bar z - m_{KF}|/\sigma_{KF}`` and the spread error is
``|\hat\sigma/\sigma_{KF} - 1|``.

```@example val
function errors_vs_kalman(estimator; seed)
    twin = Twin(model, sensors; estimator, prior = MvNormal(m0, P0), process_noise = q,
        rng = Xoshiro(seed))
    em, es = Float64[], Float64[]
    for (k, obs) in enumerate(data)
        step!(twin, obs)
        append!(em, abs.(state(twin) .- kf_mean[k]) ./ kf_sd[k])
        append!(es, abs.(uncertainty(twin) ./ kf_sd[k] .- 1))
    end
    return mean(em), mean(es)
end
avg(est) = Tuple(mean(collect(errors_vs_kalman(est; seed = s)) for s in 1:3))

Ns = [50, 200, 800, 3200]
v1 = DataFrame(N = Int[], EnKF_mean_err = Float64[], EnKF_sd_err = Float64[],
    PF_mean_err = Float64[], PF_sd_err = Float64[])
for N in Ns
    push!(v1, (N, avg(EnKF(N = N))..., avg(ParticleFilter(N = N))...))
end
v1
```

```@example val
slope(y) = (X = [ones(length(Ns)) log.(Ns)]; (X \ log.(y))[2])
slopes = (EnKF = slope(v1.EnKF_mean_err), PF = slope(v1.PF_mean_err))
```

```@example val
[verdict!("V1 EnKF convergence rate", "slope in [-0.65, -0.35]",
     string(round(slopes.EnKF; digits = 2)), -0.65 ≤ slopes.EnKF ≤ -0.35),
 verdict!("V1 PF convergence rate", "slope in [-0.65, -0.35]",
     string(round(slopes.PF; digits = 2)), -0.65 ≤ slopes.PF ≤ -0.35),
 verdict!("V1 EnKF mean at N = 3200", "< 0.05 σ",
     string(round(v1.EnKF_mean_err[end]; digits = 3), " σ"), v1.EnKF_mean_err[end] < 0.05),
 verdict!("V1 EnKF spread at N = 3200", "< 3%",
     string(round(100v1.EnKF_sd_err[end]; digits = 1), "%"), v1.EnKF_sd_err[end] < 0.03)]
```

The slopes are close to the theoretical −1/2. Quadrupling the members halves the error,
which is exactly what Monte Carlo theory predicts. At equal ``N``, the particle filter
is noisier than the EnKF. Reweighting and resampling waste members, whereas the EnKF
moves all of them.

## [V2. Joint parameter estimation is exact Bayesian inference](@id val-param)

**Theory.** Augmenting the state with unknown parameters (see
[Unknown parameters: the augmented state](@ref)) turns parameter estimation into
filtering [Kitagawa1998](@cite). Consider a quantity drifting at an unknown rate:
``\dot u = \theta``, observed with noise. The augmented state ``z = (u, \theta)``
evolves linearly:

```math
z_k = \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix} z_{k-1},
```

so the Kalman filter on ``z`` gives the exact posterior of ``\theta``. This is Bayesian
linear regression of ``y`` on ``t``.

**Acceptance criterion.** With ``N = 4000``, the EnKF posterior mean of ``\theta`` is
within 0.2 posterior standard deviations of the exact value, and its standard deviation
is within 10% of it.

```@example val
drift(u, p, t) = [p[1]]
dmodel = TwinModel(drift, [0.0], [0.0]; states = [:u], params = [:θ],
    abstol = 1e-10, reltol = 1e-10)
dsensors = ObservationModel(dmodel, [:u], 0.2)
ddata = let rng = Xoshiro(2)
    [Observation(t, [0.3 * t + 0.2 * randn(rng)]) for t in 1.0:1.0:20.0]
end
dprior = MvNormal([0.0, 0.0], Diagonal([0.1^2, 1.0]))
dk_mean, dk_sd = kalman([1.0 1.0; 0.0 1.0], zeros(2, 2), [1.0 0.0], fill(0.04, 1, 1),
    mean(dprior), Matrix(cov(dprior)), ddata)
exact_θ = (mean = dk_mean[end][2], sd = dk_sd[end][2])

v2 = map([EnKF(N = 4000), ParticleFilter(N = 4000)]) do est
    twin = Twin(dmodel, dsensors; estimator = est, estimate = [:θ], prior = dprior,
        rng = Xoshiro(3))
    foreach(o -> step!(twin, o), ddata)
    (estimator = string(nameof(typeof(est))), θ_mean = twin[:θ], θ_sd = uncertainty(twin, :θ))
end
push!(v2, (estimator = "exact (Kalman)", θ_mean = exact_θ.mean, θ_sd = exact_θ.sd))
DataFrame(v2)
```

```@example val
enkf_θ = v2[1]
verdict!("V2 EnKF parameter posterior", "mean < 0.2 σ, sd within 10%",
    string(round(abs(enkf_θ.θ_mean - exact_θ.mean) / exact_θ.sd; digits = 2), " σ, ",
        round(100 * (enkf_θ.θ_sd / exact_θ.sd - 1); digits = 1), "%"),
    abs(enkf_θ.θ_mean - exact_θ.mean) < 0.2exact_θ.sd &&
        abs(enkf_θ.θ_sd / exact_θ.sd - 1) < 0.1)
```

The EnKF reproduces the exact posterior. The particle filter's mean is also correct,
but its standard deviation is too large. This is not a bug but a known limitation, and
the check makes it visible.

A static parameter has no dynamics, so resampling can only *copy* particles, never
create new values. Early observations are weighed against a fixed set of parameter
values, and the cloud cannot refine below their spacing (*sample impoverishment*).
This is why [`ParticleFilter`](@ref) has a `jitter` option, and why
[Joint state–parameter estimation in practice](@ref) recommends the EnKF or artificial
parameter noise [LiuWest2001](@cite).

## [V3. The particle filter is exact where the EnKF cannot be](@id val-nonlinear)

**Theory.** For nonlinear, non-Gaussian problems the EnKF is only an approximation,
because its update is linear (see [The ensemble Kalman filter (EnKF)](@ref)). The
particle filter has no such restriction. We test both on a hard case: a decaying
quantity observed through its **square**,

```math
\dot u = -a u, \qquad y_k = u_k^2 + \varepsilon_k .
```

The observation cannot tell ``+u`` from ``-u``, so the posterior has two modes. In one
dimension the exact posterior can be computed on a fine grid, by evaluating the
filtering recursion (see [The filtering recursion](@ref)) numerically
[Kitagawa1987](@cite). With 1201 grid points the grid error is negligible.

**Acceptance criteria.**

1. The particle filter's error decreases from ``N = 1000`` to ``N = 20000``.
2. At ``N = 20000``, its probability of the positive mode is within 0.05 of the grid
   filter's at every step.
3. At ``N = 20000``, its posterior standard deviation is within 0.05 of the grid
   filter's at every step.

The EnKF is shown for comparison only. It is *expected* to fail here.

```@example val
a, qn, σn = 0.1, 0.3, 0.5
decay(u, p, t) = [-p[1] * u[1]]
nmodel = TwinModel(decay, [0.0], [a]; states = [:u], params = [:a],
    abstol = 1e-10, reltol = 1e-10)
nsensors = ObservationModel((u, p, t) -> [u[1]^2], [σn]; names = [:y])
nprior = Normal(0.3, 1.0)
ndata = let rng = Xoshiro(4), u = -1.2               # the truth is in the smaller mode
    map(1:15) do k
        u = exp(-a) * u + qn * randn(rng)
        Observation(Float64(k), [u^2 + σn * randn(rng)])
    end
end

# Reference: grid-based Bayesian filter
grid = collect(range(-6, 6; length = 1201))
transition = [pdf(Normal(exp(-a) * g0, qn), g1) for g1 in grid, g0 in grid]
summarize(x, w) = (m = sum(w .* x);
    (P_pos = sum(w[x .> 0]), mean = m, sd = sqrt(sum(w .* (x .- m) .^ 2))))
grid_stats = let p = pdf.(nprior, grid) ./ sum(pdf.(nprior, grid))
    map(ndata) do obs
        p = transition * p                                   # predict
        p = p .* pdf.(Normal(0, σn), obs.y[1] .- grid .^ 2)  # update (Bayes' rule)
        p ./= sum(p)
        summarize(grid, p)
    end
end

function run_nonlinear(est)
    twin = Twin(nmodel, nsensors; estimator = est, prior = MvNormal([0.3], Diagonal([1.0])),
        process_noise = qn, rng = Xoshiro(5))
    map(ndata) do obs
        step!(twin, obs)
        b = twin.belief
        w = b isa ParticleBelief ? exp.(b.logw .- maximum(b.logw)) : ones(size(b.X, 2))
        summarize(vec(b.X), w ./ sum(w))
    end
end
maxerr(st, f) = maximum(abs(f(s) - f(g)) for (s, g) in zip(st, grid_stats))

v3 = DataFrame(method = String[], P_pos = Float64[], mean = Float64[], sd = Float64[],
    max_err_P_pos = Float64[], max_err_sd = Float64[])
for (name, est) in [("PF, N = 1000", ParticleFilter(N = 1000)),
                    ("PF, N = 20000", ParticleFilter(N = 20000)),
                    ("EnKF, N = 1000", EnKF(N = 1000))]
    st = run_nonlinear(est)
    push!(v3, (name, st[end]..., maxerr(st, s -> s.P_pos), maxerr(st, s -> s.sd)))
end
push!(v3, ("exact (grid)", grid_stats[end]..., 0.0, 0.0))
v3
```

```@example val
[verdict!("V3 PF converges to exact posterior", "error decreases with N",
     "$(round(v3.max_err_P_pos[1]; digits = 3)) → $(round(v3.max_err_P_pos[2]; digits = 3))",
     v3.max_err_P_pos[2] < v3.max_err_P_pos[1]),
 verdict!("V3 PF mode probability, N = 20000", "< 0.05 at every step",
     string(round(v3.max_err_P_pos[2]; digits = 3)), v3.max_err_P_pos[2] < 0.05),
 verdict!("V3 PF spread, N = 20000", "< 0.05 at every step",
     string(round(v3.max_err_sd[2]; digits = 3)), v3.max_err_sd[2] < 0.05)]
```

The particle filter recovers both modes and their relative weight. The EnKF's linear
update cannot represent a two-mode posterior, and its spread is badly wrong, as the
theory predicts. This is why the [estimator guide](@ref "Choosing an estimator: summary")
recommends the particle filter for small, strongly nonlinear or multimodal problems.

## [V4. Forecasts and scenarios propagate each member exactly](@id val-forecast)

**Theory.** A [`forecast`](@ref) pushes each member through the model without noise.
A [`scenario`](@ref) does the same, with interventions applied at their times. For the
linear oscillator both are exact matrix maps. Changing ``\omega`` to 1.5 two time units
ahead gives

```math
u(T + 5) = e^{A(1.5)\cdot 3}\, e^{A(1.0)\cdot 2}\, u(T).
```

**Acceptance criterion.** Every sample matches its exact value to ``10^{-8}``, limited
only by the solver tolerance.

```@example val
twin = Twin(model, sensors; estimator = EnKF(N = 500), prior = MvNormal(m0, P0),
    process_noise = q, rng = Xoshiro(6))
foreach(o -> step!(twin, o), data)
Z, T = members(twin), twin.t

fc = forecast(twin, T + 5; saveat = [T + 5])
sc = scenario(twin, SetParameter(T + 2, :ω => 1.5), T + 5; saveat = [T + 5])
err_fc = maximum(abs.(fc.samples[:, end, :] .- exp(A(1.0, 0.1) * 5) * Z))
err_sc = maximum(abs.(sc.samples[:, end, :] .- exp(A(1.5, 0.1) * 3) * exp(A(1.0, 0.1) * 2) * Z))
[verdict!("V4 forecast", "max error < 1e-8", string(round(err_fc; sigdigits = 2)), err_fc < 1e-8),
 verdict!("V4 scenario with SetParameter", "max error < 1e-8",
     string(round(err_sc; sigdigits = 2)), err_sc < 1e-8)]
```

## [V5. Calibration finds the maximum-likelihood estimate](@id val-calibrate)

**Theory.** [`calibrate!`](@ref) maximizes the Gaussian likelihood of the data (see
[Calibration: maximum likelihood](@ref)). For the drift model with ``u(0) = 0``, the
prediction is ``\theta t``. Maximizing the likelihood is then least squares through the
origin, which has the closed form

```math
\hat\theta = \frac{\sum_k t_k y_k}{\sum_k t_k^2}.
```

**Acceptance criterion.** `calibrate!` agrees with the closed form to a relative error
of ``10^{-4}``, the optimizer's tolerance.

```@example val
ctwin = Twin(dmodel, dsensors; estimator = NoAssimilation(N = 1),
    prior = MvNormal([0.0], Diagonal([1e-12])))
calibrate!(ctwin, ddata, [:θ]; alg = NelderMead())
θ_fit = Dict(paramvalues(ctwin))[:θ]
θ_exact = sum(o.t * o.y[1] for o in ddata) / sum(o.t^2 for o in ddata)
rel = abs(θ_fit / θ_exact - 1)
(θ_fit = θ_fit, θ_exact = θ_exact,
 verdict = verdict!("V5 calibrate! = closed-form MLE", "relative error < 1e-4",
     string(round(rel; sigdigits = 2)), rel < 1e-4))
```

## [V6. Statistical calibration: is the uncertainty honest?](@id val-sbc)

The checks so far compare single runs. A stronger test asks whether Vardr's
uncertainty is right *on average* over many systems.

**Theory.** If the "real" system is drawn from exactly the prior and noise model the
twin assumes, then the true state is a draw from the twin's posterior. This is the
*perfect-model* setting. Three consequences can be tested
[CookGelmanRubin2006, TaltsEtAl2018](@cite):

- A 90% credible interval contains the truth in 90% of experiments (**coverage**).
- The rank of the truth among the members is **uniform**
  [Anderson1996, Hamill2001](@cite).
- The normalized innovation squared (NIS) averages 1 (see
  [Without ground truth (real data): innovations](@ref)).

We run ``M = 200`` independent experiments with 20 observations each.

**Acceptance criteria.**

1. Coverage of the 50% and 90% intervals is within the 95% binomial range for
   ``M = 200`` (about ±7 and ±4 percentage points).
2. A χ² test of the 10-bin rank histogram does not reject uniformity (``p > 0.01``).
3. The mean NIS is between 0.9 and 1.1.

```@example val
function perfect_model_experiments(M; N = 200)
    hits50, hits90, ranks, nis = 0, 0, Int[], Float64[]
    for r in 1:M
        rng = Xoshiro(1000 + r)
        x = m0 .+ sqrt.(diag(P0)) .* randn(rng, 2)          # a system drawn from the prior
        twin = Twin(model, sensors; estimator = EnKF(N = N), prior = MvNormal(m0, P0),
            process_noise = q, rng = Xoshiro(r))
        for k in 1:20
            x = F * x .+ sqrt.(diag(Q)) .* randn(rng, 2)
            y = x[1] + 0.1 * randn(rng)
            predict!(twin, k * Δt)
            push!(nis, (y - twin[:x])^2 / (uncertainty(twin, :x)^2 + 0.1^2))
            assimilate!(twin, Observation(k * Δt, [y]))
        end
        xs = members(twin)[1, :]
        hits50 += quantile(xs, 0.25) < x[1] < quantile(xs, 0.75)
        hits90 += quantile(xs, 0.05) < x[1] < quantile(xs, 0.95)
        push!(ranks, count(<(x[1]), xs))
    end
    return (cov50 = hits50 / M, cov90 = hits90 / M, ranks = ranks, nis = mean(nis))
end
M = 200
sbc = perfect_model_experiments(M)
counts = [count(r -> 20b ≤ r < 20(b + 1) || (b == 9 && r == 200), sbc.ranks) for b in 0:9]
χ2 = sum((counts .- M / 10) .^ 2 ./ (M / 10))
p_value = ccdf(Chisq(9), χ2)
(coverage_50 = sbc.cov50, coverage_90 = sbc.cov90, mean_NIS = round(sbc.nis; digits = 3),
 rank_histogram = counts, p_value = round(p_value; digits = 2))
```

```@example val
binom_ok(c, p) = abs(c - p) ≤ 1.96 * sqrt(p * (1 - p) / M)
[verdict!("V6 coverage of 50% intervals", "within binomial 95% range",
     string(sbc.cov50), binom_ok(sbc.cov50, 0.5)),
 verdict!("V6 coverage of 90% intervals", "within binomial 95% range",
     string(sbc.cov90), binom_ok(sbc.cov90, 0.9)),
 verdict!("V6 rank histogram uniform", "χ² p > 0.01", string(round(p_value; digits = 2)),
     p_value > 0.01),
 verdict!("V6 mean NIS", "in [0.9, 1.1]", string(round(sbc.nis; digits = 3)),
     0.9 ≤ sbc.nis ≤ 1.1)]
```

The twin's uncertainty is honest. Its intervals contain the truth as often as they
claim to, and its innovations have the size it predicts. This is the property that
makes forecast probabilities, such as the overload risk in the
[bioprocess case](@ref "Case study: a bioprocess twin"), meaningful.

## [Validation summary](@id val-summary)

```@example val
verdicts
```

```@example val
all(verdicts.pass) ? "All $(nrow(verdicts)) checks passed." :
    "$(count(!, verdicts.pass)) of $(nrow(verdicts)) checks FAILED."
```

**What this establishes.**

- On problems with known answers, Vardr's estimators, forecasts, scenarios and
  calibration compute what the theory says, to within Monte Carlo error that shrinks at
  the expected rate.
- Its uncertainty statements are statistically calibrated in the perfect-model setting.

**What it does not establish.**

- That a particular *model* is adequate for a particular system. That is validation of
  your twin, done with your data, following the [Step-by-step workflow](@ref).
- Performance outside the regimes tested, such as very high dimensions, other solvers or
  stiff models. Repeat the relevant checks for your own configuration. The functions on
  this page are a template.

**Known limitations confirmed.**

- The EnKF is approximate for strongly non-Gaussian posteriors (V3).
- The particle filter overstates the uncertainty of static parameters unless they are
  given artificial noise or jitter (V2).
- Both are documented in [Bayesian filtering and data assimilation](@ref).

**Reproducing.** Every number on this page comes from code with fixed random seeds. The
same checks, at smaller sizes, run in the test suite: `julia --project -e 'using Pkg;
Pkg.test()'`.
