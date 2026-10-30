# Vardr.jl examples

Each script is self-contained. It simulates a "real system" with known ground truth,
builds a twin from imperfect prior knowledge, and reports how well the twin recovers the
truth.

Set up the environment once (from the repository root):

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run any example with threads enabled:

```sh
julia --project=examples -t auto examples/lorenz63_twin.jl
```

| Example | System | What it demonstrates |
|---|---|---|
| [`oscillator_twin.jl`](oscillator_twin.jl) | Damped oscillator, 2 states | The basics: joint state–parameter estimation with the EnKF, forecast, a what-if `scenario`, `compare` |
| [`sir_epidemic_twin.jl`](sir_epidemic_twin.jl) | SIR epidemic (ModelingToolkit) | MTK models with symbolic indexing, replaying data with `TableSource` + `run!`, intervention scenarios |
| [`lorenz63_twin.jl`](lorenz63_twin.jl) | Lorenz-63, chaotic | Keeping a chaotic system synchronized from a single observed variable; open-loop vs assimilated error; forecast skill and ensemble spread as a predictability estimate |
| [`pkpd_patient_twin.jl`](pkpd_patient_twin.jl) | Two-compartment PK + effect-site Emax PD | A personalized medical twin: repeated dosing via a model callback, sparse and mixed measurements (`NaN` = not measured), nonlinear `h(u, p, t)` that depends on estimated parameters, comparing dosing regimens for efficacy and toxicity |
| [`bioreactor_fedbatch_twin.jl`](bioreactor_fedbatch_twin.jl) | Fed-batch *E. coli* bioreactor making a recombinant protein | A pharmaceutical process twin: capacitance, off-gas and load-cell sensors every 30 min plus lab assays every 4 h (`NaN` between), an oxygen-uptake observation that depends on estimated parameters, learning a hidden pump under-delivery, and choosing a feed cap that keeps the risk of oxygen overload below 5% |
| [`thermal_room_twin.jl`](thermal_room_twin.jl) | 2R2C building model with thermostat | Live operation: a sensor thread feeding `ChannelSource`, sinks that forecast and raise alerts, `SnapshotSink`, `WallClock` through a sensor outage with rollback of provisional ticks, querying the twin while it runs, retrofit what-ifs |
| [`heat_rod_twin.jl`](heat_rod_twin.jl) | 1-D heat equation, 40 cells | A high-dimensional (PDE) twin with fewer ensemble members than unknowns; covariance localization vs an unlocalized EnKF and a 300-member reference |
| [`tank_leak_detection_twin.jl`](tank_leak_detection_twin.jl) | Three-tank process (ModelingToolkit) | Fault detection: tracking an unmeasured tank and a leak coefficient that changes abruptly, a credible-interval alarm, and operator decision support (throughput vs overflow risk) |
| [`hybrid_bioreactor_twin.jl`](hybrid_bioreactor_twin.jl) | Batch bioreactor with substrate-inhibited growth | A hybrid `NeuralODEModel`: known mass balances plus a network for the unknown growth rate, trained with gradients (`calibrate!` + ForwardDiff); the trained model as a twin of a new batch, with the network's output bias fine-tuned online; forecasting when the substrate runs out |
| [`agent_epidemic_twin.jl`](agent_epidemic_twin.jl) | Agent-based epidemic: 5000 people, 3 age groups, contact matrix | An `AgentModel` with a `ParticleFilter`: fixed 5-day infections, learning the transmission rate and reporting fraction from reported cases and hospital admissions, `Event` interventions that change agents' contacts (school closure vs shielding the elderly) |
