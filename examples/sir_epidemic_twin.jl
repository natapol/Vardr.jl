# A digital twin of an epidemic (SIR model written with ModelingToolkit).
#
#   julia --project=examples -t auto examples/sir_epidemic_twin.jl
#
# The twin assimilates daily prevalence data, estimates the transmission rate β online,
# and compares an intervention scenario against business as usual.

using Vardr, ModelingToolkit, Random
using SciMLBase: remake, solve
using OrdinaryDiffEqTsit5: Tsit5
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters β γ
@variables S(t) I(t) R(t)
@mtkcompile sir = System([D(S) ~ -β * S * I, D(I) ~ β * S * I - γ * I, D(R) ~ γ * I], t)
prob = ODEProblem(sir, [S => 0.99, I => 0.01, R => 0.0, β => 0.5, γ => 0.1], (0.0, 1.0))

# Synthetic "surveillance data": the true β is 0.4
truth = solve(remake(prob; p = [β => 0.4], tspan = (0.0, 40.0)), Tsit5(); saveat = 1.0:1.0:40.0)
rng = Xoshiro(7)
data = [Observation(day, truth[I][k] + 0.002 * randn(rng)) for (k, day) in enumerate(truth.t)]

model = TwinModel(prob)
twin = Twin(model, ObservationModel(model, [I], 0.002);
    name = "epidemic", estimator = EnKF(N = 200), estimate = [β],
    process_noise = [0.0, 0.0, 0.0, 0.002], bounds = (zeros(4), fill(Inf, 4)),
    rng = Xoshiro(8))

# Replay the data as if it were a live feed
runner = run!(twin, TableSource((t = [o.t for o in data], I = [o.y[1] for o in data]));
    sinks = [CallbackSink((tw, obs) -> obs.t % 10 == 0 &&
        println("day $(obs.t): β ≈ $(round(tw[β]; digits = 3))"))])
println(runner)

business_as_usual = forecast(twin, 120.0; name = "business as usual")
distancing = scenario(twin, SetParameter(45.0, β => 0.1), 120.0; name = "distancing from day 45")
display(compare(business_as_usual, distancing; var = R))
