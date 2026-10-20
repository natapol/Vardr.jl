obs = synthetic_observations()
a = oscillator_twin()
for o in obs[1:50]
    step!(a, o)
end
path = joinpath(mktempdir(), "twin.jld2")
snapshot(path, a)
for o in obs[51:end]
    step!(a, o)
end

b = oscillator_twin(; rng = Xoshiro(99))
restore!(b, path)
@test b.t == obs[50].t
@test length(history(b)) == 1 + 2 * 50
for o in obs[51:end]
    step!(b, o)
end
@test members(b) == members(a)
@test history(b).mean == history(a).mean

@test_throws DimensionMismatch restore!(Twin(oscillator_model()), path)
@test_throws ArgumentError restore!(oscillator_twin(; estimator = ParticleFilter(N = 100)), path)
