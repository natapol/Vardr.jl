m = oscillator_model()
@test nstates(m) == 2
@test statenames(m) == [:x, :v]
@test paramnames(m) == [:k, :c]
@test Vardr.state_index(m, :v) == 2
@test Vardr.state_index(m, 1) == 1
@test_throws ArgumentError Vardr.state_index(m, :nope)

# Unnamed models get default names
mu = TwinModel(osc, [1.0, 0.0], [4.0, 0.1])
@test statenames(mu) == [:u1, :u2]
@test isempty(paramnames(mu))

# Observation models
om = ObservationModel(m, [:x], 0.1)
@test om.names == [:x]
@test om.h([2.0, 3.0], nothing, 0.0) == [2.0]
@test om.R ≈ fill(0.01, 1, 1)
energy = ObservationModel((u, p, t) -> [u[1]^2 + u[2]^2], [0.2]; names = [:E])
@test energy.names == [:E]
@test_throws ArgumentError ObservationModel((u, p, t) -> u, [1.0 2.0; 3.0 4.0])
@test Observation(1, 2).y == [2.0]

# Twin construction and open-loop prediction
tw = Twin(m, om; estimator = NoAssimilation(N = 20), rng = Xoshiro(0))
@test tw.t == 0.0
@test size(members(tw)) == (2, 20)
predict!(tw, 1.0)
@test tw.t == 1.0
@test length(history(tw)) == 2
@test_throws ArgumentError predict!(tw, 0.5)
@test tw[:x] == state(tw)[1]
@test Dict(paramvalues(tw)) == Dict(:k => 4.0, :c => 0.1)

# A twin with a degenerate prior follows the deterministic model exactly
z0 = repeat([1.0, 0.0], 1, 5)
det = Twin(m; estimator = NoAssimilation(N = 5), prior = z0)
predict!(det, 2.0)
@test state(det) ≈ Vardr.propagate(m, [1.0, 0.0], [4.0, 0.1], 0.0, 2.0)
@test all(iszero, uncertainty(det))

# Bounds are enforced on members
bounded = Twin(m; estimator = NoAssimilation(N = 50), bounds = ([0.0, -Inf], [Inf, Inf]),
    rng = Xoshiro(0))
@test all(≥(0), members(bounded)[1, :])

# History is a Tables.jl table
cols = Vardr.Tables.columns(history(tw))
@test keys(cols) == (:t, :kind, :x, :x_std, :v, :v_std)
@test cols.kind == [:init, :predict]

# Callbacks and tstops given to the model (e.g. a dosing schedule) are kept alongside
# Vardr's own callbacks.
decay(u, p, t) = [-p[1] * u[1]]
dose_times = [1.0, 2.0]
dosing = Vardr.DiscreteCallback((u, t, integ) -> t in dose_times, integ -> (integ.u[1] += 1.0))
md = TwinModel(decay, [0.0], [0.5]; states = [:A], params = [:k], callback = dosing,
    tstops = dose_times)
after_doses = exp(-0.5 * 1.5) + exp(-0.5 * 0.5)  # amount at t = 2.5
@test Vardr.propagate(md, [0.0], [0.5], 0.0, 2.5)[1] ≈ after_doses rtol = 1e-4
dtw = Twin(md; estimator = NoAssimilation(N = 2), prior = zeros(1, 2))
fcd = forecast(dtw, 2.5; saveat = [1.0, 2.0, 2.5])
@test fcd[:A][:, 1] ≈ [1.0, 1 + exp(-0.5), after_doses] rtol = 1e-4  # post-dose values
# ...and combined with intervention callbacks
fce = scenario(dtw, Event(1.5, integ -> (integ.u[1] = 0.0)), 2.5; saveat = [2.5])
@test fce[:A][1, 1] ≈ exp(-0.5 * 0.5) rtol = 1e-4
