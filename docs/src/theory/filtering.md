```@meta
CurrentModule = Vardr
```

# Bayesian filtering and data assimilation

This chapter derives the algorithms that keep a twin synchronized with data. The
notation follows [Digital twins and state-space models](@ref): the augmented state is
``z = [u; \theta]`` and the observations are ``y_k = h(z_k) + \varepsilon_k`` with
``\varepsilon_k \sim \mathcal N(0, R)``.

## The filtering recursion

The goal is the filtering distribution ``p(z_k \mid y_{1:k})``. Because the model is
Markov (the next state depends only on the current one) and each observation depends
only on the current state, it can be computed **recursively**, one observation at a time
[Jazwinski1970, Sarkka2013](@cite).

**Prediction** (Chapman–Kolmogorov equation): push the current belief through the
dynamics.

```math
p(z_k \mid y_{1:k-1}) = \int p(z_k \mid z_{k-1}) \, p(z_{k-1} \mid y_{1:k-1}) \, \mathrm{d}z_{k-1}
```

**Update** (Bayes' rule): weigh the prediction by how well it explains the new
measurement.

```math
p(z_k \mid y_{1:k}) =
\frac{p(y_k \mid z_k)\, p(z_k \mid y_{1:k-1})}{p(y_k \mid y_{1:k-1})},
\qquad
p(y_k \mid z_k) = \mathcal N\big(y_k;\ h(z_k),\ R\big).
```

The prediction is also called the *forecast* or *prior*, and the update the *analysis*.
Prediction spreads uncertainty, because of chaos and model error; the update contracts
it.

!!! note "In Vardr"
    [`predict!`](@ref)`(twin, t)` is the prediction step and
    [`assimilate!`](@ref)`(twin, obs)` the update step. [`step!`](@ref) does both. All
    estimators share the prediction step: every member is integrated with the ODE
    solver, in parallel over threads, and then model error is added:
    ```math
    z_k^{(i)} = \begin{bmatrix}\mathcal M(u_{k-1}^{(i)}, \theta_{k-1}^{(i)}) \\ \theta_{k-1}^{(i)}\end{bmatrix}
    + \sqrt{\Delta t_k}\; q \odot \xi^{(i)}, \qquad \xi^{(i)} \sim \mathcal N(0, I).
    ```
    The estimators differ only in the update.

## The Kalman filter

If the dynamics and ``h`` are linear and all distributions are Gaussian, the recursion
has an exact closed form, the **Kalman filter** [Kalman1960](@cite). With the predicted
mean ``z^f`` and covariance ``P^f``, and ``h(z) = Hz``, the update is

```math
\begin{aligned}
K &= P^f H^\top \big(H P^f H^\top + R\big)^{-1} &&\text{(Kalman gain)}\\
z^a &= z^f + K\,\big(y - H z^f\big) &&\text{(analysis mean)}\\
P^a &= (I - K H)\, P^f &&\text{(analysis covariance)}
\end{aligned}
```

The gain ``K`` weighs model against data. A confident model (small ``P^f``) moves little;
a confident sensor (small ``R``) pulls the estimate towards the measurement. Note that
``P^f H^\top`` is the covariance between the state and the predicted observation, and
``HP^fH^\top`` is the covariance of the predicted observation:

```math
K = \operatorname{Cov}(z, h(z))\, \big[\operatorname{Cov}(h(z)) + R\big]^{-1}.
```

This form needs no matrix ``H`` and no explicit ``P^f``, only covariances, which can be
estimated from samples. That is the key idea of the ensemble Kalman filter.

## The ensemble Kalman filter (EnKF)

The EnKF [Evensen1994, Evensen2007](@cite) represents the predicted distribution
by ``N`` equally weighted members ``z^{(i)}``. It estimates the covariances in the
Kalman gain from the ensemble, and applies the Kalman update to every member. The
model can be nonlinear, because each member is propagated through the full nonlinear
model. The Gaussian assumption is made *only in the update*.

### The algorithm Vardr implements

Given the predicted members ``Z = [z^{(1)}, \dots, z^{(N)}]`` and an observation ``y``
with covariance ``R``:

1. **Inflation** (optional, factor ``\lambda \ge 1``):
   ``z^{(i)} \leftarrow \bar z + \lambda\,(z^{(i)} - \bar z)``.
2. **Predicted observations**: ``\hat y^{(i)} = h(u^{(i)}, \theta^{(i)}, t_k)``.
   Missing entries (`NaN`) of ``y`` are removed, together with the matching rows of
   ``\hat y`` and rows and columns of ``R``.
3. **Anomalies and sample covariances**:
   ```math
   A = Z - \bar z\,\mathbf 1^\top, \quad
   B = \hat Y - \bar{\hat y}\,\mathbf 1^\top, \quad
   C_{zy} = \frac{A B^\top}{N-1}, \quad
   C_{yy} = \frac{B B^\top}{N-1}.
   ```
4. **Perturbed observations** [BurgersEtAl1998, HoutekamerMitchell1998](@cite): each
   member sees its own noisy copy of the measurement,
   ``y^{(i)} = y + \varepsilon^{(i)}``, with ``\varepsilon^{(i)} \sim \mathcal N(0, R)``.
5. **Update every member**:
   ```math
   z^{(i)} \leftarrow z^{(i)} + C_{zy}\,\big(C_{yy} + R\big)^{-1}\big(y^{(i)} - \hat y^{(i)}\big).
   ```

Why perturb the observations? Without step 4, every member would move by the same
deterministic amount, and the analysis spread would be too small. With perturbed
observations, the analysis ensemble has the correct Kalman covariance ``(I - KH)P^f``
in expectation [BurgersEtAl1998](@cite).

### Inflation

With finite ``N``, sampling errors make the EnKF systematically *underestimate* its own
uncertainty. The spread shrinks a little too much at every update until the filter
ignores the data. **Multiplicative inflation** [AndersonAnderson1999](@cite) counteracts
this by spreading the members about their mean before each update. Typical values are
1.01–1.10. Model error (`process_noise`) acts similarly, but it is additive and applied
during prediction.

### Localization

With fewer members than unknowns, sample covariances contain **spurious long-range
correlations**. In a 40-cell heat model with 10 members, a sensor at one end appears
correlated with the other end purely by chance, and the update corrupts distant cells.
**Covariance localization** [HoutekamerMitchell2001, HamillWhitakerSnyder2001](@cite)
multiplies covariances elementwise (a *Schur product*, ``\circ``) by a taper that decays
with distance:

```math
K_u = \big(L_{uy} \circ C_{uy}\big)\big(L_{yy} \circ C_{yy} + R\big)^{-1}.
```

Here ``L_{uy}`` tapers state–observation pairs and ``L_{yy}`` observation–observation
pairs. A common taper is the compactly supported, fifth-order piecewise rational
function of [GaspariCohn1999](@cite), which looks like a Gaussian but is exactly zero
beyond twice its length scale.

Both tapers must come from the same correlation function. The gain is then the exact
Kalman gain of the localized covariance ``L \circ P^f``, which is positive semi-definite
by the Schur product theorem. Tapering only ``C_{uy}`` is a common shortcut, but it can
increase the spread at the sensors and make the filter diverge.

Estimated parameters ``\theta`` are global (they affect every location), so they have
no meaningful "distance". Vardr updates them with the *unlocalized* gain
``C_{\theta y}(C_{yy} + R)^{-1}``.

!!! note "In Vardr"
    `EnKF(N, inflation = λ, localization = (Lxy, Lyy))`. If the observation model
    selects states directly, `Lyy` is the rows of `Lxy` at the observed states, so you
    can pass `Lxy` alone. See `examples/heat_rod_twin.jl` for a Gaspari–Cohn
    implementation. There, localization cuts the error by a factor of three with only 10
    members.

### Strengths and limits of the EnKF

- **Cost**: ``N`` model runs per step, independent of the state dimension. It scales to
  very large models, especially with localization.
- **Robustness**: members are moved, not reweighted, so the EnKF does not degenerate.
- **Gaussian update**: the update is linear in the innovation. For strongly nonlinear
  ``h``, or for bimodal or skewed posteriors, the EnKF is approximate. It is still
  usually accurate when observations are frequent compared with the system's nonlinear
  time scales.
- **Bounds**: the linear update can push members across physical limits. Vardr clips
  them to `bounds` afterwards.

## The particle filter

The **bootstrap particle filter** [GordonSalmondSmith1993](@cite) makes no Gaussian
assumption. It represents the belief by weighted particles and applies Bayes' rule
literally.

1. **Predict** each particle through the model (as above).
2. **Weight** each particle by the likelihood of the observation:
   ```math
   \log w^{(i)} \leftarrow \log w^{(i)} + \log \mathcal N\big(y;\ h(z^{(i)}),\ R\big).
   ```
3. **Resample** when the weights become uneven. The **effective sample size**
   ```math
   \mathrm{ESS} = \frac{1}{\sum_i \tilde w_i^2}, \qquad \tilde w_i = \frac{w_i}{\sum_j w_j}
   ```
   is ``N`` for equal weights and 1 when a single particle holds all the weight
   [DoucetJohansen2011](@cite). If ``\mathrm{ESS} < \tau N``, Vardr draws ``N`` new
   particles with **systematic resampling** [Kitagawa1996](@cite), which uses one random
   number and has low variance, and resets the weights to equal.
4. **Jitter** (optional): after resampling, duplicated particles are identical. Adding
   noise ``\sigma \odot j\,\xi``, where ``\sigma`` is the posterior standard deviation,
   restores diversity. This matters most for parameters, which have no dynamics of their
   own to spread them out [LiuWest2001](@cite).

The particle filter converges to the exact Bayesian solution as ``N \to \infty``, for any
nonlinearity. The price is the **curse of dimensionality**: the number of particles
needed grows exponentially with the effective dimension of the problem
[SnyderEtAl2008](@cite). Use it for small problems (≲ 10 unknowns) with strong
nonlinearity or multimodality, and the EnKF otherwise.

!!! note "In Vardr"
    `ParticleFilter(N, resample_threshold = τ, jitter = j)`. If every particle has zero
    likelihood (`all particle weights are zero`), the observation is incompatible with
    the whole belief. Widen the prior, add process noise, or check `R`.

## Joint state–parameter estimation in practice

With the augmented state, the update of a parameter is driven by the sample
correlation between that parameter and the predicted observations. In EnKF terms:

```math
\Delta\theta^{(i)} = C_{\theta y}\,(C_{yy} + R)^{-1}\,\big(y^{(i)} - \hat y^{(i)}\big).
```

Practical consequences [Evensen2009csm, MoradkhaniEtAl2005](@cite):

- **Identifiability.** If changing ``\theta`` barely changes the measured outputs,
  ``C_{\theta y} \approx 0`` and the parameter is never updated. Its uncertainty stays at
  the prior. This is a property of the model and sensors, not of the filter. See
  [Identifiability](@ref).
- **Delayed sensitivity.** Some parameters only affect outputs after some time, or under
  certain conditions (excitation). A heater efficiency is invisible while the heater is
  off. Learning then happens in bursts.
- **Collapse.** Repeated updates shrink the parameter spread. Without artificial evolution
  (`process_noise` on ``\theta``) the spread eventually reaches zero, and the twin can no
  longer track changes.
- **Abrupt changes** (a valve fails, a leak opens) need larger parameter noise, or a
  detection rule on the innovations. See `examples/tank_leak_detection_twin.jl`.

## Choosing an estimator: summary

| Situation | Recommended |
|:---|:---|
| Default, any size, mildly nonlinear | `EnKF(N = 50–200, inflation = 1.01–1.05)` |
| Many states (PDEs, networks), ``N`` ≪ dimension | `EnKF` with `localization` |
| ≲ 10 unknowns, strongly nonlinear or multimodal | `ParticleFilter(N = 1000+, jitter = 0.02–0.1)` |
| Simulation only, or a baseline | `NoAssimilation` |
