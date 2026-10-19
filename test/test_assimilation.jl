obs = synthetic_observations()
xT = true_state(10.0)

for est in (EnKF(N = 100), EnKF(N = 100, inflation = 1.02),
    ParticleFilter(N = 2000), ParticleFilter(N = 2000, jitter = 0.1))
    @testset "$(nameof(typeof(est)))" begin
        tw = oscillator_twin(; estimator = est)
        open_loop = oscillator_twin(; estimator = NoAssimilation(N = 100))
        for o in obs
            step!(tw, o)
        end
        predict!(open_loop, 10.0)
        err = norm(state(tw) - xT)
        @test err < 0.1
        @test err < norm(state(open_loop) - xT) / 5
        # Joint estimation recovers the spring constant k = 3
        @test abs(tw[:k] - TRUE_P[1]) < 0.1
        @test uncertainty(tw, :k) < 0.2

        # The true x lies in the 95% band at (roughly) the nominal rate
        h = history(tw)
        upd = findall(==(:update), h.kind)
        inside = count(upd) do i
            abs(h.mean[i][1] - true_state(h.t[i])[1]) ≤ 1.96 * h.std[i][1]
        end
        # (a particle filter without jitter is somewhat overconfident: sample impoverishment)
        @test inside / length(upd) > (est isa EnKF ? 0.8 : 0.6)
        @test length(h.observations) == length(obs)
    end
end

@testset "partial observations" begin
    m = oscillator_model()
    om = ObservationModel(m, [:x, :v], [0.05, 0.05])
    tw = Twin(m, om; estimator = EnKF(N = 50), rng = Xoshiro(3),
        prior = MvNormal([1.0, 0.0], Diagonal([0.2, 0.2] .^ 2)))
    v_before = uncertainty(tw, :v)
    step!(tw, Observation(0.0, [NaN, 0.0]))  # only v is measured
    @test uncertainty(tw, :v) < v_before
    n = length(history(tw))
    step!(tw, Observation(0.5, [NaN, NaN]))  # nothing measured: predict only
    @test history(tw).kind[end] == :predict
    @test length(history(tw)) == n + 1
end

@testset "errors" begin
    tw = oscillator_twin()
    @test_throws DimensionMismatch assimilate!(tw, Observation(0.0, [1.0, 2.0]))
    @test_throws ArgumentError assimilate!(tw, Observation(1.0, [1.0]))
    @test_throws ArgumentError assimilate!(Twin(oscillator_model()), Observation(0.0, [1.0]))
    @test_throws ArgumentError EnKF(N = 1)
    @test_throws ArgumentError ParticleFilter(resample_threshold = 2)
end

@testset "systematic resampling" begin
    idx = Vardr.systematic_resample(Xoshiro(0), [0.0, 1.0, 0.0, 0.0], 10)
    @test all(==(2), idx)
    idx = Vardr.systematic_resample(Xoshiro(0), fill(0.25, 4), 4)
    @test sort(idx) == 1:4
end

@testset "localization" begin
    m = oscillator_model()
    om = ObservationModel(m, [:x], OBS_σ)
    prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2))
    mk(est) = Twin(m, om; estimator = est, estimate = [:k], prior, rng = Xoshiro(4))
    obs = Observation(0.0, [1.2])

    # An all-ones taper is the same as no localization
    a, b = mk(EnKF(N = 30)), mk(EnKF(N = 30, localization = ones(2, 1)))
    assimilate!(a, obs)
    assimilate!(b, obs)
    @test members(a) ≈ members(b)

    # Estimated parameters always get the unlocalized gain; tapered states change
    a, b = mk(EnKF(N = 30)), mk(EnKF(N = 30, localization = [1.0; 0.0;;]))
    assimilate!(a, obs)
    assimilate!(b, obs)
    @test members(a)[3, :] ≈ members(b)[3, :]
    @test members(b)[2, :] == members(mk(EnKF(N = 30)))[2, :]  # v fully tapered: untouched

    # Lyy is derived for direct state observations; general h needs both matrices
    @test Vardr.localization_pair(ones(2, 1), om)[2] == ones(1, 1)
    energy = ObservationModel((u, p, t) -> [u[1]^2 + u[2]^2], [0.1])
    @test_throws ArgumentError Vardr.localization_pair(ones(2, 1), energy)

    # Regression: a consistent (Lxy, Lyy) taper never inflates the spread at the
    # sensors, however small the ensemble
    n = 30
    xs = range(0, 1; length = n)
    rw(u, p, t) = zero(u)
    mr = TwinModel(rw, zeros(n), [0.0])
    sensors = collect(1:3:n)
    taper = [exp(-((xs[i] - xs[j]) / 0.1)^2) for i in 1:n, j in sensors]
    tw = Twin(mr, ObservationModel(mr, sensors, 0.1); prior = randn(Xoshiro(0), n, 6),
        estimator = EnKF(N = 6, localization = taper), rng = Xoshiro(1))
    before = uncertainty(tw)[sensors]
    assimilate!(tw, Observation(0.0, zeros(length(sensors))))
    @test all(uncertainty(tw)[sensors] .<= before .+ 1e-12)
end
