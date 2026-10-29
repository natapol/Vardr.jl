# A hybrid (neural ODE) twin of a batch bioreactor.
#
#   julia --project=examples -t auto examples/hybrid_bioreactor_twin.jl
#
# The mass balances of a batch culture are known, but the specific growth rate μ(S) is
# not: the real organism is inhibited by high substrate concentrations (Haldane
# kinetics), which a Monod model cannot describe. A hybrid model keeps the mass
# balances and lets a small neural network learn μ(S) from one training batch. The
# trained model then becomes the twin of a new batch, where the EnKF corrects what the
# network got wrong, and the twin forecasts when the substrate runs out.

using Vardr, Random, Statistics, LinearAlgebra, Distributions, Printf
using ForwardDiff: ForwardDiff
using Optimization: AutoForwardDiff
using OptimizationOptimJL: BFGS

# --- The real organism (unknown to the modeller) --------------------------------------
μ_true(S) = 0.8 * S / (2.0 + S + S^2 / 15.0)       # g/L substrate → 1/h
Y_true = 0.5                                       # g biomass per g substrate
batch(u, p, t) = (μ = μ_true(max(u[2], 0.0)); [μ * u[1], -μ * u[1] / p[1]])
plant = ODEModel(batch, [0.1, 20.0], [Y_true]; states = [:X, :S], params = [:Y])
measure(rng, u0, ts, σ) = [Observation(t,
    Vardr.propagate(plant, u0, [Y_true], 0.0, t) .+ σ .* randn(rng, 2)) for t in ts]

# One training batch, sampled every 30 min: biomass (X) and glucose (S), g/L
σ = [0.05, 0.3]
training = measure(Xoshiro(1), [0.1, 20.0], 0.5:0.5:14.0, σ)

# --- Hybrid model: known mass balances, learned growth rate ---------------------------
# Physical knowledge built into the hybrid: 0 ≤ μ ≤ 1 1/h, and no growth without
# substrate. The network sees S scaled to order 1.
net = MLP(1, 8, 1)
μ_hybrid(S, nn) = 1 / (1 + exp(-nn([S / 10])[1])) * S / (S + 0.05)
hybrid(u, p, t, nn) = (μ = μ_hybrid(max(u[2], 0.0), nn); [μ * u[1], -μ * u[1] / p[1]])
model = NeuralODEModel(net, [0.1, 20.0]; f = hybrid, p = [0.5], params = [:Y],
    states = [:X, :S], rng = Xoshiro(2), maxiters = 10_000)
println(model)

# Train the weights by maximum likelihood (gradients by forward-mode AD)
sensors = ObservationModel(model, [:X, :S], σ)
trainer = Twin(model, sensors; estimator = NoAssimilation(N = 1),
    prior = reshape([0.1, 20.0], 2, 1))
fit = calibrate!(trainer, training, weightnames(model);
    alg = BFGS(initial_stepnorm = 0.01), adtype = AutoForwardDiff(), u0 = [0.1, 20.0])
@printf("training: negative log-likelihood %.1f for %d readings\n", fit.objective,
    2length(training))

w = networkweights(model, trainer.p)
println("\n  S (g/L)   μ learned   μ true")
for S in (0.5, 2.0, 5.0, 10.0, 15.0, 20.0)
    @printf("  %6.1f   %8.3f   %7.3f\n", S, μ_hybrid(S, x -> net(x, w)), μ_true(S))
end

# --- Twin of a new batch ------------------------------------------------------------
# Different start: 0.2 g/L biomass, 15 g/L glucose. Biomass is read every 30 min by an
# optical density probe; glucose is assayed at-line every 2 h (NaN in between).
u0 = [0.2, 15.0]
rng = Xoshiro(3)
readings = map(measure(rng, u0, 0.5:0.5:12.0, σ)) do o
    Observation(o.t, [o.y[1], o.t % 2 == 0 ? o.y[2] : NaN])
end

# The trained weights become the twin's parameters. The EnKF tracks the states and
# also fine-tunes the network's output bias online (the last weight), which shifts
# the learned growth rate up or down for this batch.
twin_model = NeuralODEModel(net, u0; f = hybrid, p = [0.5], params = [:Y],
    states = [:X, :S], weights = w, maxiters = 10_000)
bias = last(weightnames(twin_model))
twin = Twin(twin_model, ObservationModel(twin_model, [:X, :S], σ);
    estimator = EnKF(N = 100), estimate = [bias],
    prior = MvNormal(vcat(u0, w[end]), Diagonal([0.05, 1.0, 0.5] .^ 2)),
    process_noise = [0.05, 0.2, 0.05], bounds = ([0.0, 0.0, -Inf], fill(Inf, 3)),
    rng = Xoshiro(4))
open_loop = Twin(twin_model; estimator = NoAssimilation(N = 100),
    prior = MvNormal(u0, Diagonal([0.05, 1.0] .^ 2)), rng = Xoshiro(4))

t_now = 5.0
for o in readings
    o.t <= t_now && step!(twin, o)
end
predict!(open_loop, t_now)

# When will the glucose run out (S < 0.5 g/L)?
depletion(fc) = [something(findfirst(<(0.5), fc[:S][:, k]), length(fc.t)) for k in 1:size(fc.samples, 3)]
ts = collect(t_now:0.1:12.0)
truth_S = [Vardr.propagate(plant, u0, [Y_true], 0.0, t)[2] for t in ts]
t_true = ts[findfirst(<(0.5), truth_S)]
println("\nAt t = $t_now h:")
for (name, tw) in (("open-loop hybrid", open_loop), ("hybrid twin (EnKF)", twin))
    fc = forecast(tw, 12.0; saveat = ts)
    td = ts[depletion(fc)]
    @printf("  %-19s X = %5.2f g/L, S = %5.2f g/L; glucose runs out at %.1f h (90%%: %.1f–%.1f)\n",
        name, mean(fc[:X][1, :]), mean(fc[:S][1, :]), mean(td), quantile(td, 0.05),
        quantile(td, 0.95))
end
u_true = Vardr.propagate(plant, u0, [Y_true], 0.0, t_now)
@printf("  %-19s X = %5.2f g/L, S = %5.2f g/L; glucose runs out at %.1f h\n", "truth",
    u_true..., t_true)
