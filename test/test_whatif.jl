m = oscillator_model()
z0 = repeat([1.0, 0.0], 1, 4)
tw = Twin(m; estimator = NoAssimilation(N = 4), prior = z0)

fc = forecast(tw, 5.0; saveat = 0:0.5:5)
@test fc.t == collect(0:0.5:5)
@test size(fc.samples) == (2, 11, 4)
@test mean(fc)[:, end] ≈ Vardr.propagate(m, [1.0, 0.0], [4.0, 0.1], 0.0, 5.0) rtol = 1e-4
@test size(fc[:x]) == (11, 4)
@test all(quantile(fc, 0.05) .≤ quantile(fc, 0.95))
@test tw.t == 0.0  # forecasting does not move the twin
@test_throws ArgumentError forecast(tw, 0.0)

# A parameter intervention matches a two-stage direct solve
sc = scenario(tw, [SetParameter(2.0, :k => 1.0)], 5.0; saveat = [5.0], name = "soft")
u2 = Vardr.propagate(m, [1.0, 0.0], [4.0, 0.1], 0.0, 2.0)
u5 = Vardr.propagate(m, u2, [1.0, 0.1], 2.0, 5.0)
@test sc.samples[:, end, 1] ≈ u5 rtol = 1e-4
@test sc.name == "soft"

# Interventions at the start, state resets and generic events
sc0 = scenario(tw, SetParameter(0.0, :k => 1.0), 5.0; saveat = [5.0])
@test sc0.samples[:, end, 1] ≈ Vardr.propagate(m, [1.0, 0.0], [1.0, 0.1], 0.0, 5.0) rtol = 1e-4
reset = scenario(tw, SetState(3.0, :x => 0.0, :v => 0.0), 5.0; saveat = [5.0])
@test reset.samples[:, end, 1] ≈ [0.0, 0.0] atol = 1e-8
stop_ = scenario(tw, Event(3.0, integ -> (integ.u .= 0)), 5.0; saveat = [5.0])
@test stop_.samples[:, end, 1] ≈ [0.0, 0.0] atol = 1e-8
@test_throws ArgumentError scenario(tw, Event(0.0, identity), 5.0)

cmp = compare(fc, sc; var = :x)
@test cmp.scenario == ["forecast", "soft"]
@test cmp.mean[2] ≈ u5[1] rtol = 1e-4

# Estimated-parameter uncertainty propagates into forecasts
twk = oscillator_twin(; estimator = EnKF(N = 50))
fck = forecast(twk, 5.0)
@test std(fck)[1, end] > 0.05
