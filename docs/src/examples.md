# Examples

The repository's
[`examples/`](https://github.com/natapol/Vardr.jl/tree/main/examples) directory contains
complete, runnable twins. Each script simulates a "real system" with known ground truth,
builds a twin from imperfect prior knowledge, and reports how well the twin recovers the
truth. They are a good template for your own twin.

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples -t auto examples/thermal_room_twin.jl
```

| Example | System | Features shown | Guide pages |
|:---|:---|:---|:---|
| `oscillator_twin.jl` | Damped oscillator | Joint state–parameter EnKF, forecast, scenario, compare | [Tutorial](@ref "Tutorial: your first twin") |
| `sir_epidemic_twin.jl` | SIR epidemic (ModelingToolkit) | Symbolic models, `TableSource` replay, interventions | [Models and sensors](@ref), [Running live](@ref) |
| `lorenz63_twin.jl` | Lorenz-63 (chaotic) | Synchronizing from one observed variable; forecast skill vs ensemble spread | [State and parameter estimation](@ref), [Forecasts and what-if scenarios](@ref) |
| `pkpd_patient_twin.jl` | Two-compartment PK/PD | Dosing callbacks, sparse mixed measurements (`NaN`), nonlinear `h` with estimated parameters, regimen comparison | [Models and sensors](@ref), [Forecasts and what-if scenarios](@ref) |
| `bioreactor_fedbatch_twin.jl` | Fed-batch *E. coli* bioreactor (biopharma) | Online + sparse lab sensors, parameter-dependent nonlinear `h` (oxygen uptake), learning a hidden pump fault, chance-constrained decision with per-sample parameters | [Case study](@ref "Case study: a bioprocess twin") |
| `thermal_room_twin.jl` | 2R2C building with a thermostat | Live `ChannelSource`, alert sinks, `SnapshotSink`, `WallClock` outages with rollback, retrofit what-ifs | [Running live](@ref) |
| `heat_rod_twin.jl` | 1-D heat equation, 40 cells | High-dimensional twin, covariance localization | [Localization](@ref guide-localization) |
| `tank_leak_detection_twin.jl` | Three-tank process (ModelingToolkit) | Fault detection, unmeasured states, abrupt parameter change, decision support | [State and parameter estimation](@ref), [Forecasts and what-if scenarios](@ref) |
| `hybrid_bioreactor_twin.jl` | Batch bioreactor with unknown growth kinetics | [`NeuralODEModel`](@ref) as a hybrid model, gradient-based training with `calibrate!`, fine-tuning a network weight online | [Neural and hybrid ODE models](@ref guide-neural) |
| `agent_epidemic_twin.jl` | Agent-based epidemic, 5000 people in 3 age groups | [`AgentModel`](@ref) with a particle filter, two observed streams, `Event` interventions on agents | [Agent-based models](@ref guide-agents) |

For checks of Vardr itself against exact answers, which you can reuse as templates for
validating your own configuration, see
[Case study: validating Vardr against exact answers](@ref).

## Building your own twin: a recipe

A short version of the [Step-by-step workflow](@ref):

1. **Model.** Write `f(u, p, t)`, or an MTK system, with named states and parameters.
   Simulate it and check that it behaves plausibly.
2. **Sensors.** List what is measured, at what noise level. Mark gaps with `NaN`.
3. **Synthetic test.** Choose "true" parameters that differ from your design values,
   simulate noisy data, and check that the twin recovers the truth. Compare it against
   `NoAssimilation`.
4. **Tune.** Set the prior, `process_noise`, `N` and `inflation` until the errors are
   small and the uncertainty is honest (~95% of the truth inside ±2σ).
5. **Calibrate** on historical data with [`calibrate!`](@ref), if you have it.
6. **Replay** history through [`run!`](@ref) with a [`TableSource`](@ref), with sinks.
7. **Go live** with a [`ChannelSource`](@ref) or your own source, a
   [`WallClock`](@ref) and a [`SnapshotSink`](@ref).
8. **Use it.** Add forecasts, alerts and what-if analyses for the decisions that matter.
