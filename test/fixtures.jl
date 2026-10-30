using Random, Statistics, LinearAlgebra, Distributions

# Damped oscillator: x'' = -k x - c x'
osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]

const TRUE_U0 = [1.5, 0.0]
const TRUE_P = [3.0, 0.1]
const OBS_σ = 0.05
const OBS_T = collect(0.1:0.1:10.0)

oscillator_model(; u0 = [1.0, 0.0], p = [4.0, 0.1]) =
    ODEModel(osc, u0, p; states = [:x, :v], params = [:k, :c])

const TRUTH = oscillator_model(u0 = TRUE_U0, p = TRUE_P)
true_state(t) = Vardr.propagate(TRUTH, TRUE_U0, TRUE_P, 0.0, t)

"Noisy observations of x from the true oscillator."
function synthetic_observations(; rng = Xoshiro(1), ts = OBS_T)
    [Observation(t, true_state(t)[1] + OBS_σ * randn(rng)) for t in ts]
end

"Twin of the oscillator observing x, estimating k, with a prior that covers the truth."
function oscillator_twin(; estimator = EnKF(N = 100), rng = Xoshiro(2), kwargs...)
    m = oscillator_model()
    om = ObservationModel(m, [:x], OBS_σ)
    prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2))
    Twin(m, om; estimator, estimate = [:k], prior, process_noise = [0.01, 0.01, 0.01],
        rng, kwargs...)
end
