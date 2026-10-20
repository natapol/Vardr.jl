# Fault detection with a digital twin: finding a leak in a three-tank process.
#
#   julia --project=examples -t auto examples/tank_leak_detection_twin.jl
#
# Three tanks in series (a classic process-control benchmark), written with
# ModelingToolkit. A pump fills tank 1, water flows 1 → 2 → 3 through valves (Torricelli
# flow), and tank 3 drains. Level sensors are installed on tanks 1 and 3 only.
#
# At t = 60 min a leak opens in tank 2, which has no sensor. The twin estimates a leak
# coefficient online, modelled as a slowly drifting random walk so it can follow
# sudden changes. An alarm fires when the leak is credibly above zero. A what-if
# analysis then shows whether the operator can keep production running by raising the
# pump rate until maintenance arrives.

using Vardr, ModelingToolkit, Distributions, LinearAlgebra, Random, Statistics, Printf
using ModelingToolkit: t_nounits as t, D_nounits as D

flow(Δh) = sign(Δh) * sqrt(abs(Δh) + 1e-6)   # Torricelli, regularized near zero

@parameters A = 1.0 k12 = 0.6 k23 = 0.6 k3 = 0.5 q_pump = 0.5 leak = 0.0
@variables h1(t) h2(t) h3(t)
pump(t) = 1 + 0.2 * sin(2π * t / 45)          # known pump demand profile
eqs = [A * D(h1) ~ q_pump * pump(t) - k12 * flow(h1 - h2),
    A * D(h2) ~ k12 * flow(h1 - h2) - k23 * flow(h2 - h3) - leak * flow(h2),
    A * D(h3) ~ k23 * flow(h2 - h3) - k3 * flow(h3)]
@mtkcompile tanks = System(eqs, t)
prob = ODEProblem(tanks, [h1 => 1.2, h2 => 0.85, h3 => 0.6], (0.0, 1.0))
model = TwinModel(prob)

# --- The real plant: simulated with Vardr itself (a twin with no uncertainty) ----------
# The leak opens at t = 60. Each sample of a scenario is one plant trajectory.
tobs = 1.0:1.0:180.0
plant = Twin(model; estimator = NoAssimilation(N = 2), prior = repeat(prob.u0, 1, 2))
real_run = scenario(plant, SetParameter(60.0, leak => 0.08), 180.0; saveat = tobs,
    nsamples = 1)
rng = Xoshiro(17)
level(name, k) = real_run[name][k, 1]
data = [Observation(tk, [level(:h1, k), level(:h3, k)] .+ 0.005 .* randn(rng, 2))
        for (k, tk) in enumerate(tobs)]

# --- The twin: estimate the levels and the leak coefficient --------------------------------
sensors = ObservationModel(model, [h1, h3], 0.005)
n = length(prob.u0)
prior = MvNormal(vcat(prob.u0, 0.0), Diagonal(vcat(fill(0.05^2, n), 0.01^2)))
twin = Twin(model, sensors; name = "tank-line-3", estimator = EnKF(N = 100, inflation = 1.01),
    estimate = [leak], prior, rng = Xoshiro(18),
    process_noise = vcat(fill(0.002, n), 0.004),   # leak can change over time
    bounds = (vcat(zeros(n), 0.0), fill(Inf, n + 1)))

alarm = nothing
println("  t    leak estimate      h2 estimate (unmeasured)   h2 true")
for (k, obs) in enumerate(data)
    step!(twin, obs)
    z, s = twin[leak], uncertainty(twin, leak)
    if alarm === nothing && z - 3s > 0.01
        global alarm = obs.t
        @printf("%5.0f  🚨 LEAK ALARM: leak = %.3f ± %.3f (opened at t = 60)\n", obs.t, z, s)
    end
    if obs.t % 20 == 0
        @printf("%5.0f   %.3f ± %.3f        %.3f ± %.3f               %.3f\n", obs.t, z, s,
            twin[h2], uncertainty(twin, h2), level(:h2, k))
    end
end
@printf("\nDetection delay: %.0f min. Estimated leak %.3f ± %.3f (true 0.080)\n",
    alarm - 60, twin[leak], uncertainty(twin, leak))

# --- What-if: keep production on target until maintenance arrives in 2 hours? -----------
# The leak loses water before it reaches tank 3, so the plant delivers less product.
# Could the operator raise the pump to compensate, without overflowing tank 1 (3.5 m tall)?
t0 = twin.t
horizon = t0 + 120
saveat = t0:1.0:horizon
outflow(h3) = 0.5 * flow(h3)                       # k3 · √h3, product delivered (m³/min)
delivered(fc) = vec(sum(outflow.(fc[:h3][2:end, :]); dims = 1))  # m³ over 2 h, per sample
target = delivered(scenario(twin, SetParameter(t0, leak => 0.0), horizon; saveat,
    name = "no leak"))
options = [forecast(twin, horizon; saveat, name = "do nothing"),
    scenario(twin, SetParameter(t0, q_pump => 0.55), horizon; saveat, name = "pump +10%"),
    scenario(twin, SetParameter(t0, q_pump => 0.6), horizon; saveat, name = "pump +20%"),
    scenario(twin, SetParameter(t0, q_pump => 0.7), horizon; saveat, name = "pump +40%"),
    scenario(twin, [SetParameter(t0 + 30, leak => 0.0)], horizon; saveat,
        name = "patch the leak in 30 min")]
@printf("\nTarget without the leak: %.1f m³ in the next 2 h\n", mean(target))
println("Option                      delivered (m³)   P(on target)   P(tank 1 overflows)")
for fc in options
    d = delivered(fc)
    p_overflow = mean(maximum(fc[:h1]; dims = 1) .> 3.5)
    @printf("%-28s %8.1f ± %.1f %12.0f%% %18.0f%%\n", fc.name, mean(d), std(d),
        100mean(d .>= 0.98 .* target), 100p_overflow)
end
