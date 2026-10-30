# Neural ODE and agent-based models
using ForwardDiff: ForwardDiff
using Optimization: AutoForwardDiff
using OptimizationOptimJL: BFGS, NelderMead

@testset "MLP" begin
    net = MLP(2, 3, 1; activation = tanh)
    @test nweights(net) == 3 * 2 + 3 + 1 * 3 + 1
    w = collect(range(-1, 1; length = nweights(net)))
    W1, b1 = reshape(w[1:6], 3, 2), w[7:9]
    W2, b2 = reshape(w[10:12], 1, 3), w[13:13]
    x = [0.3, -0.7]
    @test net(x, w) ≈ W2 * tanh.(W1 * x .+ b1) .+ b2
    @test_throws DimensionMismatch net(x, w[1:5])
    @test_throws ArgumentError MLP(3)

    w0 = init_weights(Xoshiro(1), net; output_scale = 0.0)
    @test length(w0) == nweights(net)
    @test all(iszero, w0[10:end])              # last layer switched off
    @test ForwardDiff.gradient(w -> sum(net(x, w)), w) isa Vector{Float64}
end

@testset "NeuralODEModel" begin
    net = MLP(2, 4, 2)
    w = init_weights(Xoshiro(2), net; output_scale = 1.0)
    m = NeuralODEModel(net, [1.0, 0.5]; weights = w, states = [:a, :b])
    @test nstates(m) == 2 && statenames(m) == [:a, :b]
    @test nweights(m) == nweights(net)
    @test weightnames(m) == [Symbol(:nn, i) for i in 1:nweights(net)]
    @test networkweights(m) == w
    ref = ODEModel((u, p, t) -> net(u, w), [1.0, 0.5], Float64[])
    @test Vardr.propagate(m, [1.0, 0.5], m.ode.prob.p, 0.0, 2.0) ≈
          Vardr.propagate(ref, [1.0, 0.5], Float64[], 0.0, 2.0)
    @test_throws ArgumentError NeuralODEModel(net, [1.0, 0.5]; p = [1.0], params = [:nn1])
    @test_throws ArgumentError NeuralODEModel(net, [1.0, 0.5]; p = [1.0], params = [:a, :b])

    # Universal differential equation: logistic growth with an unknown crowding term
    logistic(u, p, t) = [p[1] * u[1] * (1 - u[1] / 10)]
    truth = ODEModel(logistic, [0.5], [1.0])
    data = [Observation(t, Vardr.propagate(truth, [0.5], [1.0], 0.0, t)[1])
            for t in 0.5:0.5:8.0]
    crowding = MLP(1, 6, 1)
    ude = NeuralODEModel(crowding, [0.5]; p = [1.0], params = [:r], states = [:N],
        f = (u, p, t, nn) -> [p[1] * u[1] * (1 + nn([u[1] / 10])[1])], rng = Xoshiro(3),
        maxiters = 10_000)
    tw = Twin(ude, ObservationModel(ude, [:N], 0.05); estimator = NoAssimilation(N = 1),
        prior = fill(0.5, 1, 1))
    # A small first step keeps the optimizer away from exploding trajectories
    sol = calibrate!(tw, data, weightnames(ude); alg = BFGS(initial_stepnorm = 0.01),
        adtype = AutoForwardDiff(), u0 = [0.5])
    learned = networkweights(ude, tw.p)
    @test crowding([0.5], learned)[1] ≈ -0.5 atol = 0.05      # true term: -N/10
    fc = forecast(tw, 8.0; saveat = [8.0])
    @test fc.samples[1, end, 1] ≈ Vardr.propagate(truth, [0.5], [1.0], 0.0, 8.0)[1] rtol = 0.02

    # Mechanistic parameters keep working in scenarios, and weights can be estimated
    slow = scenario(tw, SetParameter(0.0, :r => 0.0), 8.0; saveat = [8.0])
    @test slow.samples[1, end, 1] ≈ 0.5
    online = Twin(ude, ObservationModel(ude, [:N], 0.05); estimator = EnKF(N = 20),
        estimate = weightnames(ude), rng = Xoshiro(4))
    foreach(o -> step!(online, o), data[1:3])
    @test online.t == data[3].t
    @test isfinite(online[:nn1])
end

# Individual-based SIR epidemic
mutable struct Village
    status::Vector{Int8}          # 0 susceptible, 1 infectious, 2 recovered
    new_cases::Int
end
const VILLAGE_N = 500
function village(u, p, rng)
    s = zeros(Int8, VILLAGE_N)
    s[1:clamp(round(Int, u[2]), 1, VILLAGE_N)] .= 1
    Village(s, 0)
end
function village_step!(v, p, t, rng)
    β, γ = p
    pinf = 1 - exp(-β * count(==(1), v.status) / length(v.status))
    new = 0
    for i in eachindex(v.status)
        if v.status[i] == 0
            rand(rng) < pinf && (v.status[i] = 1; new += 1)
        elseif v.status[i] == 1
            rand(rng) < γ && (v.status[i] = 2)
        end
    end
    v.new_cases = new
    return v
end
village_counts(v) = [count(==(0), v.status), count(==(1), v.status),
    count(==(2), v.status), v.new_cases]
# Reassign statuses to match (rounded) compartment counts
function village_lift(v, u, p, rng)
    nI = clamp(round(Int, u[2]), 0, VILLAGE_N)
    nR = clamp(round(Int, u[3]), 0, VILLAGE_N - nI)
    status = shuffle(rng, vcat(fill(Int8(1), nI), fill(Int8(2), nR),
        fill(Int8(0), VILLAGE_N - nI - nR)))
    Village(status, v.new_cases)
end
village_model(; lift = nothing) = AgentModel(village, village_step!, village_counts,
    [490, 10, 0, 0], [0.5, 0.2]; states = [:S, :I, :R, :new], params = [:β, :γ], lift)

@testset "AgentModel" begin
    m = village_model()
    @test nstates(m) == 4 && paramnames(m) == [:β, :γ]
    @test_throws ArgumentError AgentModel(village, village_step!, village_counts,
        [1, 2], [0.5]; states = [:a])
    @test_throws ArgumentError AgentModel(village, village_step!, village_counts,
        [1, 2], [0.5]; dt = 0)

    truth = village([490, 10, 0, 0], [0.6, 0.2], Xoshiro(7))
    rng = Xoshiro(8)
    data = map(1:30) do d
        village_step!(truth, [0.6, 0.2], d - 1, rng)
        Observation(d, 0.6 * truth.new_cases + 2 * randn(rng))
    end
    reported = ObservationModel((u, p, t) -> [0.6 * u[4]], [2.0]; names = [:reported])
    prior = MvNormal([490, 10, 0, 0, 0.4], Diagonal([1, 3, 1, 1, 0.15] .^ 2))
    pf_twin(seed) = Twin(m, reported; estimator = ParticleFilter(N = 300, jitter = 0.1),
        estimate = [:β], prior, process_noise = [0, 0, 0, 0, 0.005], rng = Xoshiro(seed))

    tw = pf_twin(9)
    consistent(tw) = all(tw.belief.X[1:4, i] == village_counts(tw.belief.latent[i])
                         for i in axes(tw.belief.X, 2))
    @test length(tw.belief.latent) == 300 && consistent(tw)
    foreach(o -> step!(tw, o), data[1:15])
    @test consistent(tw)
    @test all(sum(tw.belief.X[1:3, :]; dims = 1) .== VILLAGE_N)
    # Resampled duplicates are independent copies
    @test length(unique(objectid.(tw.belief.latent))) == 300
    @test tw[:β] ≈ 0.6 atol = 0.08

    # Reproducible from the twin's rng
    tw2 = pf_twin(9)
    foreach(o -> step!(tw2, o), data[1:15])
    @test tw2.belief.X == tw.belief.X

    # Forecasts work on copies and support interventions
    before = deepcopy(tw.belief.X)
    fc = forecast(tw, 30.0; saveat = 15:30)
    @test tw.belief.X == before && consistent(tw)
    @test size(fc.samples) == (4, 16, 300)
    @test all(sum(fc.samples[1:3, :, :]; dims = 1) .== VILLAGE_N)
    lockdown = scenario(tw, SetParameter(20.0, :β => 0.0), 30.0; saveat = [30.0])
    @test all(iszero, lockdown[:new][end, :])
    vaccinate = scenario(tw, Event(20.0, (sim, p, t) -> replace!(sim.status, 0 => 2)), 30.0;
        saveat = [30.0])
    @test all(iszero, vaccinate[:S][end, :])
    @test_throws ArgumentError scenario(tw, SetState(20.0, :S => 0), 30.0)
    @test_throws ArgumentError predict!(tw, tw.t + 0.5)        # off the step grid

    # State rows are summaries: no noise or bounds on them, and the EnKF needs `lift`
    @test_throws ArgumentError Twin(m, reported; estimator = ParticleFilter(N = 10),
        process_noise = 1.0)
    @test_throws ArgumentError Twin(m, reported; estimator = ParticleFilter(N = 10),
        bounds = (zeros(4), fill(Inf, 4)))
    @test_throws ArgumentError Twin(m, reported; estimator = EnKF(N = 10))
    lifted = village_model(; lift = village_lift)
    en = Twin(lifted, ObservationModel(lifted, [:I], 3.0); estimator = EnKF(N = 50),
        prior = MvNormal([490, 10, 0, 0], Diagonal([1, 3, 1, 1] .^ 2)), rng = Xoshiro(10))
    foreach(d -> step!(en, Observation(d, 10.0 + 4d)), 1:5)
    @test consistent(en)

    # Snapshots keep the agents
    path = tempname() * ".jld2"
    snapshot(path, tw)
    tw3 = pf_twin(1)
    restore!(tw3, path)
    step!(tw, data[16])
    step!(tw3, data[16])
    @test tw3.belief.X == tw.belief.X

    # Calibration with common random numbers
    ct = Twin(m, reported; estimator = NoAssimilation(N = 1),
        prior = reshape([490.0, 10, 0, 0], 4, 1), rng = Xoshiro(1))
    calibrate!(ct, data, [:β]; alg = NelderMead(), nsims = 8)
    @test Dict(paramvalues(ct))[:β] ≈ 0.6 atol = 0.1
end
