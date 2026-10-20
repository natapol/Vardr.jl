using OptimizationOptimJL: NelderMead

obs = synthetic_observations()
m = oscillator_model()
om = ObservationModel(m, [:x], OBS_σ)
tw = Twin(m, om; estimator = NoAssimilation(N = 10), rng = Xoshiro(0))
sol = calibrate!(tw, obs, [:k, :c]; alg = NelderMead(), u0 = TRUE_U0)
@test sol.u ≈ TRUE_P rtol = 0.05
@test Dict(paramvalues(tw))[:k] ≈ TRUE_P[1] rtol = 0.05

# Calibrating an online-estimated parameter recenters its members
twk = oscillator_twin()
calibrate!(twk, obs, [:k]; alg = NelderMead(), u0 = TRUE_U0)
@test twk[:k] ≈ TRUE_P[1] rtol = 0.05

@test_throws ArgumentError calibrate!(Twin(m), obs, [:k]; alg = NelderMead())
@test_throws ArgumentError calibrate!(tw, Observation[], [:k]; alg = NelderMead())
