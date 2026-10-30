# A digital twin of a chaotic system: the Lorenz-63 attractor.
#
#   julia --project=examples -t auto examples/lorenz63_twin.jl
#
# Chaos is the hardest case for a twin: any initial error grows exponentially, so an
# open-loop simulation becomes useless within a few time units. Only x is measured
# (y and z are hidden), yet data assimilation keeps all three states locked onto the
# real system. Afterwards we measure how far ahead the synchronized twin can predict.

using Vardr, Distributions, LinearAlgebra, Random, Statistics
using SciMLBase: remake, solve
using OrdinaryDiffEqTsit5: Tsit5

lorenz(u, p, t) = [p[1] * (u[2] - u[1]),
    u[1] * (p[2] - u[3]) - u[2],
    u[1] * u[2] - p[3] * u[3]]
p_true = [10.0, 28.0, 8 / 3]  # σ, ρ, β

# --- The real system (unknown to the twin) -----------------------------------------------
model = ODEModel(lorenz, [1.0, 1.0, 1.0], p_true; states = [:x, :y, :z],
    params = [:σ, :ρ, :β])
Δt = 0.1                           # sensor sampling interval
tobs = collect(Δt:Δt:20.0)
truth = solve(remake(model.prob; u0 = [-5.9, -5.5, 24.5], tspan = (0.0, 30.0)), Tsit5();
    saveat = vcat(0.0, tobs, 20.0 .+ (0.05:0.05:10.0)), abstol = 1e-10, reltol = 1e-10)
true_state(t) = truth.u[findfirst(≈(t), truth.t)]
rng = Xoshiro(11)
σ_obs = 1.0
data = [Observation(t, true_state(t)[1] + σ_obs * randn(rng)) for t in tobs]

# --- Twin: starts far from the truth, observes x only ------------------------------------
sensors = ObservationModel(model, [:x], σ_obs)
prior = MvNormal([0.0, 0.0, 20.0], Diagonal(fill(5.0^2, 3)))
make_twin(est) = Twin(model, sensors; estimator = est, prior, rng = Xoshiro(12),
    process_noise = 0.1)
twin = make_twin(EnKF(N = 50, inflation = 1.05))
open_loop = make_twin(NoAssimilation(N = 50))

rmse(tw, t) = norm(state(tw) - true_state(t)) / √3
println("  t     RMSE (open loop)   RMSE (twin)   std(z) twin")
for obs in data
    step!(twin, obs)
    predict!(open_loop, obs.t)
    if obs.t % 2 ≈ 0 || obs.t % 2 ≈ 2
        println(lpad(round(obs.t; digits = 1), 5), lpad(round(rmse(open_loop, obs.t); digits = 2), 18),
            lpad(round(rmse(twin, obs.t); digits = 2), 14),
            lpad(round(uncertainty(twin, :z); digits = 2), 14))
    end
end

# Time-averaged error over the second half of the run (after spin-up)
late = findall(>(10), history(twin).t)
upd = filter(i -> history(twin).kind[i] == :update, late)
avg = mean(norm(history(twin).mean[i] - true_state(history(twin).t[i])) / √3 for i in upd)
println("\nMean analysis RMSE for t > 10: ", round(avg; digits = 3),
    " (sensor noise σ = $σ_obs, and y, z are never measured)")

# --- Predictability: how long does a forecast from the synchronized twin stay useful? ----
fc = forecast(twin, 30.0; saveat = 20.0:0.05:30.0)
err = [norm(mean(fc)[:, j] - true_state(t)) / √3 for (j, t) in enumerate(fc.t)]
spread = [norm(std(fc)[:, j]) / √3 for j in eachindex(fc.t)]
horizon = fc.t[something(findfirst(>(5.0), err), length(err))] - 20.0
println("Forecast stays within RMSE 5 for ", round(horizon; digits = 2),
    " time units (≈ ", round(horizon / 1.1; digits = 1), " Lyapunov times)")
j = findfirst(>(5.0), err)
j === nothing ||
    println("…and the ensemble knows it: forecast spread at that time = ",
        round(spread[j]; digits = 2), " (spread at t = 20: ", round(spread[1]; digits = 2), ")")
