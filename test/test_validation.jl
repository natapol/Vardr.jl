# Checks against exact answers (smaller versions of docs/src/cases/validation.md)
using OptimizationOptimJL: NelderMead

damped_osc(u, p, t) = [u[2], -p[1]^2 * u[1] - 2p[2] * p[1] * u[2]]
vmodel = TwinModel(damped_osc, [1.0, 0.0], [1.0, 0.1]; states = [:x, :v], params = [:ω, :ζ],
    abstol = 1e-10, reltol = 1e-10)
vsensors = ObservationModel(vmodel, [:x], 0.1)
Amat(ω, ζ) = [0.0 1.0; -ω^2 -2ζ*ω]
Δt = 0.5
F = exp(Amat(1.0, 0.1) * Δt)
q = [0.05, 0.05]
Q = Δt * Diagonal(q .^ 2)
m0, P0 = [1.0, 0.0], Diagonal([0.5, 0.5] .^ 2)
vdata = let rng = Xoshiro(1), x = m0 .+ sqrt.(diag(P0)) .* randn(rng, 2)
    map(1:20) do k
        x = F * x .+ sqrt.(diag(Q)) .* randn(rng, 2)
        Observation(k * Δt, [x[1] + 0.1 * randn(rng)])
    end
end

function kalman(F, Q, H, R, m, P, data)
    means, sds = Vector{Float64}[], Vector{Float64}[]
    for o in data
        m, P = F * m, F * P * F' + Q
        Kg = P * H' / (H * P * H' + R)
        m, P = m + Kg * (o.y - H * m), (LinearAlgebra.I - Kg * H) * P
        push!(means, m)
        push!(sds, sqrt.(diag(P)))
    end
    return means, sds
end

@testset "EnKF and particle filter converge to the Kalman filter" begin
    kf_mean, kf_sd = kalman(F, Q, [1.0 0.0], fill(0.01, 1, 1), m0, Matrix(P0), vdata)
    for (est, tol_m, tol_s) in [(EnKF(N = 2000), 0.1, 0.05), (ParticleFilter(N = 2000), 0.15, 0.1)]
        tw = Twin(vmodel, vsensors; estimator = est, prior = MvNormal(m0, P0),
            process_noise = q, rng = Xoshiro(2))
        em, es = Float64[], Float64[]
        for (k, o) in enumerate(vdata)
            step!(tw, o)
            append!(em, abs.(state(tw) .- kf_mean[k]) ./ kf_sd[k])
            append!(es, abs.(uncertainty(tw) ./ kf_sd[k] .- 1))
        end
        @test mean(em) < tol_m
        @test mean(es) < tol_s
    end
end

@testset "Augmented-state parameter posterior and calibrate! are exact" begin
    drift(u, p, t) = [p[1]]
    dm = TwinModel(drift, [0.0], [0.0]; states = [:u], params = [:θ],
        abstol = 1e-10, reltol = 1e-10)
    ds = ObservationModel(dm, [:u], 0.2)
    dd = let rng = Xoshiro(3)
        [Observation(t, [0.3 * t + 0.2 * randn(rng)]) for t in 1.0:1.0:20.0]
    end
    prior = MvNormal([0.0, 0.0], Diagonal([0.1^2, 1.0]))
    km, ks = kalman([1.0 1.0; 0.0 1.0], zeros(2, 2), [1.0 0.0], fill(0.04, 1, 1),
        mean(prior), Matrix(cov(prior)), dd)
    tw = Twin(dm, ds; estimator = EnKF(N = 2000), estimate = [:θ], prior, rng = Xoshiro(4))
    foreach(o -> step!(tw, o), dd)
    @test abs(tw[:θ] - km[end][2]) < 0.3 * ks[end][2]
    @test uncertainty(tw, :θ) ≈ ks[end][2] rtol = 0.15

    ct = Twin(dm, ds; estimator = NoAssimilation(N = 1),
        prior = MvNormal([0.0], Diagonal([1e-12])))
    calibrate!(ct, dd, [:θ]; alg = NelderMead())
    θ_exact = sum(o.t * o.y[1] for o in dd) / sum(o.t^2 for o in dd)
    @test Dict(paramvalues(ct))[:θ] ≈ θ_exact rtol = 1e-4
end

@testset "Forecasts and scenarios are exact maps of the members" begin
    tw = Twin(vmodel, vsensors; estimator = EnKF(N = 50), prior = MvNormal(m0, P0),
        process_noise = q, rng = Xoshiro(5))
    foreach(o -> step!(tw, o), vdata)
    Z, T = members(tw), tw.t
    fc = forecast(tw, T + 5; saveat = [T + 5])
    @test maximum(abs.(fc.samples[:, end, :] .- exp(Amat(1.0, 0.1) * 5) * Z)) < 1e-8
    sc = scenario(tw, SetParameter(T + 2, :ω => 1.5), T + 5; saveat = [T + 5])
    exact = exp(Amat(1.5, 0.1) * 3) * exp(Amat(1.0, 0.1) * 2) * Z
    @test maximum(abs.(sc.samples[:, end, :] .- exact)) < 1e-8
end

@testset "Particle filter matches a grid filter on a bimodal posterior" begin
    a, qn, σn = 0.1, 0.3, 0.5
    decay(u, p, t) = [-p[1] * u[1]]
    nm = TwinModel(decay, [0.0], [a]; states = [:u], params = [:a],
        abstol = 1e-10, reltol = 1e-10)
    ns = ObservationModel((u, p, t) -> [u[1]^2], [σn]; names = [:y])
    nd = let rng = Xoshiro(6), u = -1.2
        map(1:10) do k
            u = exp(-a) * u + qn * randn(rng)
            Observation(Float64(k), [u^2 + σn * randn(rng)])
        end
    end
    grid = collect(range(-6, 6; length = 601))
    T = [pdf(Normal(exp(-a) * g0, qn), g1) for g1 in grid, g0 in grid]
    p = pdf.(Normal(0.3, 1.0), grid)
    p ./= sum(p)
    tw = Twin(nm, ns; estimator = ParticleFilter(N = 10000),
        prior = MvNormal([0.3], Diagonal([1.0])), process_noise = qn, rng = Xoshiro(7))
    for o in nd
        p = T * p
        p = p .* pdf.(Normal(0, σn), o.y[1] .- grid .^ 2)
        p ./= sum(p)
        step!(tw, o)
        w = exp.(tw.belief.logw .- maximum(tw.belief.logw))
        w ./= sum(w)
        @test sum(w[vec(tw.belief.X) .> 0]) ≈ sum(p[grid .> 0]) atol = 0.06
    end
end
