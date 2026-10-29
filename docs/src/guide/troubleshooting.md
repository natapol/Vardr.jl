```@meta
CurrentModule = Vardr
```

# Troubleshooting

## The twin ignores the data (filter divergence)

**Symptoms:** the estimate drifts away from the measurements while `uncertainty` is
tiny.

The ensemble has become overconfident: with almost no spread, the update gives the data
no weight. Fixes, roughly in order:

1. Increase `process_noise` on the states. The model is never perfect, so say so.
2. Use `EnKF(inflation = 1.02)`, up to about 1.1.
3. Use more members (`N`).
4. Check the sensor noise `σ`. If it is set much smaller than the real noise, the twin
   chases noise; if much larger, it trusts data too little.

## A parameter estimate freezes or wanders

- **Freezes at a wrong value with tiny uncertainty:** the parameter ensemble collapsed.
  Give the parameter a small `process_noise`, and for a `ParticleFilter` a `jitter`.
- **Uncertainty never shrinks:** the data does not inform this parameter. It is not
  identifiable from these sensors at this stage. Measure something else, fix it at a
  calibrated value, or wait for richer data (excitation).
- **Noisy, wandering estimate:** too much `process_noise` on the parameter. Reduce it.
- **Hits a physical limit:** add `bounds`, and check that the prior covers the truth.

## "all particle weights are zero"

Every particle is incompatible with the observation. The prior or state is far from the
truth, the sensor noise is underestimated, or there is an outlier. Widen the prior,
increase `N`, add `jitter` and process noise, check `σ`, or start with an `EnKF`, which
does not have this failure mode.

## "observation at t = … but the twin is at t = …"

[`assimilate!`](@ref) needs `obs.t == twin.t`. Use [`step!`](@ref), which predicts
first. If the observation is *older* than the twin, it cannot be assimilated: sort your
data, or use [`run!`](@ref), which drops (and counts) late data.

## "cannot predict backwards"

Time only moves forward. To re-run from an earlier point, [`restore!`](@ref) a snapshot
or build a new twin.

## "IntegrationError: model integration failed with retcode …"

A member's ODE solve failed. Usually a member reached a nonphysical region (a negative
concentration in a `sqrt` or `log`), or the problem is stiff:

- add `bounds` to keep members physical;
- guard the right-hand side (`sqrt(max(h, 0))`);
- use a stiff solver: `ODEModel(prob; alg = Rodas5P())`;
- reduce an overly wide prior or `process_noise`;
- for a [`NeuralODEModel`](@ref), bound the network's output (see
  [Neural and hybrid ODE models](@ref guide-neural)).

## ModelingToolkit: states reset to their initial values

Vardr disables MTK's initialization when members are integrated. If you build problems
yourself with `remake`, pass `build_initializeprob = false` and
`initializealg = SciMLBase.NoInit()`, as [`Vardr.member_problem`](@ref) does.

## Names not found ("unknown state")

- Plain-function models need `states = [...]` and `params = [...]` to use names.
- For MTK models, both `twin[I]` and `twin[:I]` work for states. For parameters in
  `SetParameter` and `calibrate!`, pass the symbolic variable (`β`) or its `Symbol`.
- Only parameters listed in `estimate` can be read with `twin[...]`. Use
  [`paramvalues`](@ref) for the others.

## It is slow

- Start Julia with threads: `julia -t auto`.
- Use an in-place model function `f!(du, u, p, t)` for larger systems, and avoid
  allocations inside it.
- Fewer members: an EnKF with localization often works with 20–50 members, even for
  hundreds of states.
- Looser solver tolerances (`ODEModel(...; reltol = 1e-4)`). Filter noise usually
  dominates integration error.
- Forecast with fewer `nsamples`, and only the `saveat` points you need.
- The first `step!` includes compilation time; measure the second.

## Results are not reproducible

Pass `rng = Xoshiro(seed)` to `Twin` (and optionally to `forecast`). The default global
RNG is shared with everything else in the session.
