# A digital twin of a damped oscillator whose spring constant is unknown.
#
#   julia --project=examples -t auto examples/oscillator_twin.jl
#
# The "real system" is simulated with k = 3. The twin starts believing k ≈ 4, learns k from
# noisy position measurements, then forecasts and evaluates a what-if intervention.

using Vardr, Distributions, LinearAlgebra, Random

osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]

# The real system (unknown to the twin) and its sensor
real_system = TwinModel(osc, [1.5, 0.0], [3.0, 0.1]; states = [:x, :v], params = [:k, :c])
rng = Xoshiro(42)
measurements = map(0.1:0.1:10.0) do t
    x = Vardr.propagate(real_system, [1.5, 0.0], [3.0, 0.1], 0.0, t)[1]
    Observation(t, x + 0.05 * randn(rng))
end

# The twin
model = TwinModel(osc, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
twin = Twin(model, ObservationModel(model, [:x], 0.05);
    name = "oscillator", estimator = EnKF(N = 100), estimate = [:k],
    prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2)),
    process_noise = [0.01, 0.01, 0.01], rng = Xoshiro(1))

for obs in measurements
    step!(twin, obs)
end
println(twin)
println("x = ", round(twin[:x]; digits = 3), " ± ", round(uncertainty(twin, :x); digits = 3))
println("k = ", round(twin[:k]; digits = 3), " ± ", round(uncertainty(twin, :k); digits = 3),
    "   (true value 3.0)")

# What-if: add damping at t = 12
baseline = forecast(twin, 20.0)
damped = scenario(twin, [SetParameter(12.0, :c => 1.0)], 20.0; name = "extra damping")
display(compare(baseline, damped; var = :x))
