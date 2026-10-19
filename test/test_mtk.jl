using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SciMLBase: remake, solve
using OrdinaryDiffEqTsit5: Tsit5

@parameters β γ
@variables S(t) I(t) R(t)
@mtkcompile sir = System([D(S) ~ -β * S * I, D(I) ~ β * S * I - γ * I, D(R) ~ γ * I], t)
prob = ODEProblem(sir, [S => 0.99, I => 0.01, R => 0.0, β => 0.5, γ => 0.1], (0.0, 1.0))
m = TwinModel(prob)
@test Set(statenames(m)) == Set([:S, :I, :R])
@test Set(paramnames(m)) == Set([:β, :γ])

# Synthetic outbreak with β = 0.4, observed through I
truth = remake(prob; p = [β => 0.4], tspan = (0.0, 40.0))
ts = 1.0:1.0:40.0
sol = solve(truth, Tsit5(); saveat = ts)
rng = Xoshiro(5)
obs = [Observation(tk, sol[I][k] + 0.002 * randn(rng)) for (k, tk) in enumerate(ts)]

om = ObservationModel(m, [I], 0.002)
tw = Twin(m, om; estimator = EnKF(N = 100), estimate = [β],
    process_noise = [0.0, 0.0, 0.0, 0.002], bounds = (zeros(4), fill(Inf, 4)),
    rng = Xoshiro(6))
@test tw[β] ≈ 0.5 rtol = 0.1
foreach(o -> step!(tw, o), obs)
@test tw[β] ≈ 0.4 rtol = 0.05
@test tw[I] ≈ sol[I][end] rtol = 0.1
@test tw[:I] == tw[I]

base = forecast(tw, 80.0)
distancing = scenario(tw, SetParameter(45.0, β => 0.05), 80.0; name = "distancing")
c = compare(base, distancing; var = I)
@test c.mean[2] < c.mean[1]
