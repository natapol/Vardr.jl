# A spatially distributed twin: temperature along a heated rod (a 1-D PDE).
#
#   julia --project=examples -t auto examples/heat_rod_twin.jl
#
#   ∂T/∂t = α ∂²T/∂x² + q(x, t),   T(0, t) = T_left(t),   T(1, t) = 0
#
# Discretized into 40 cells, the twin has 41 unknowns: 40 temperatures plus the
# diffusivity α, which is estimated online. The real rod also feels unmodelled,
# spatially varying disturbances (drafts, contact losses), so local temperature
# features keep appearing that only nearby sensors can see.
#
# 13 thermocouples are installed, but we can afford only 10 ensemble members. With
# far fewer members than unknowns, the EnKF's sample covariances contain spurious
# long-range correlations: a sensor at one end seems to "inform" cells at the other,
# and the twin degrades. Covariance localization (`EnKF(localization = …)`) tapers
# those correlations away with distance. We compare the two with a 300-member
# reference that needs no localization.

using Vardr, Distributions, LinearAlgebra, Random, Statistics, Printf

const NX = 40
const dx = 1 / (NX + 1)
const xs = collect(range(dx, 1 - dx; length = NX))   # interior grid points

T_left(t) = 1.0 + 0.5 * sin(2π * t / 10)                            # boundary
q(x, t) = 2.0 * exp(-((x - 0.65) / 0.05)^2) * (1 + 0.5 * cos(2π * t / 7))  # heater

function rod!(dT, T, p, t)
    α = p[1]
    @inbounds for i in 1:NX
        Tl = i == 1 ? T_left(t) : T[i - 1]
        Tr = i == NX ? 0.0 : T[i + 1]
        dT[i] = α * (Tl - 2T[i] + Tr) / dx^2 + q(xs[i], t)
    end
    return nothing
end

names = [Symbol(:T, i) for i in 1:NX]
model(α, T0) = ODEModel(rod!, T0, [α]; states = names, params = [:α])

# --- The real rod ------------------------------------------------------------------------
α_true = 0.002
T0_true = 0.6 .* exp.(-((xs .- 0.3) ./ 0.15) .^ 2)   # an initial hot spot
real_rod = model(α_true, T0_true)
sensor_cells = collect(2:3:40)                       # 13 thermocouples
rng = Xoshiro(5)
tobs = 0.5:0.5:30.0
# Every 0.5 time units the real rod receives a random, spatially correlated (length
# scale ~0.05) temperature disturbance that the model knows nothing about.
Kd = [exp(-((a - b) / 0.05)^2 / 2) for a in xs, b in xs] + 1e-8I
disturbance = MvNormal(zeros(NX), Symmetric(0.06^2 .* Kd))
truth = Dict{Float64, Vector{Float64}}()
let T = T0_true, t = 0.0
    for tk in tobs
        T = Vardr.propagate(real_rod, T, [α_true], t, tk) .+ rand(rng, disturbance)
        t = tk
        truth[tk] = T
    end
end
data = [Observation(t, truth[t][sensor_cells] .+ 0.02 .* randn(rng, length(sensor_cells)))
        for t in tobs]

# --- The twin ----------------------------------------------------------------------------
# Prior: smooth random temperature profiles; α believed to be 0.0035 ± 0.0015
K = [0.3^2 * exp(-((a - b) / 0.15)^2 / 2) for a in xs, b in xs] + 1e-8I
prior = MvNormal(vcat(zeros(NX), 0.0035), Symmetric(cat(K, [0.0015^2]; dims = (1, 2))))
sensors = ObservationModel(model(0.0035, zeros(NX)), names[sensor_cells], 0.02)

# Localization: a Gaspari–Cohn taper (compactly supported, Gaussian-like) between each
# grid point and each sensor, zero beyond 0.2. The observation–observation taper is
# derived from it automatically because the sensors observe states directly, and the
# global parameter α is never localized.
function gaspari_cohn(r)
    r = abs(r)
    r >= 2 && return 0.0
    r <= 1 ? -r^5 / 4 + r^4 / 2 + 5r^3 / 8 - 5r^2 / 3 + 1 :
    r^5 / 12 - r^4 / 2 + 5r^3 / 8 + 5r^2 / 3 - 5r + 4 - 2 / (3r)
end
L = [gaspari_cohn((xs[i] - xs[j]) / 0.1) for i in 1:NX, j in sensor_cells]

function run_twin(estimator)
    twin = Twin(model(0.0035, zeros(NX)), sensors; estimator, estimate = [:α], prior,
        rng = Xoshiro(1), process_noise = vcat(fill(0.085, NX), 4e-5),
        bounds = (vcat(fill(-Inf, NX), 0.0005), fill(Inf, NX + 1)))
    unobserved = setdiff(1:NX, sensor_cells)
    err, err_between = Float64[], Float64[]
    for obs in data
        step!(twin, obs)
        d = state(twin) - truth[obs.t]
        push!(err, norm(d) / √NX)
        push!(err_between, norm(d[unobserved]) / √length(unobserved))
    end
    return twin, mean(err[31:end]), mean(err_between[31:end])  # skip the spin-up
end

println("Configuration                      RMSE   RMSE between sensors   α (true 0.002)")
for (label, est) in [("EnKF, 10 members", EnKF(N = 10, inflation = 1.05)),
    ("EnKF, 10 members + localization", EnKF(N = 10, inflation = 1.05, localization = L)),
    ("EnKF, 300 members (reference)", EnKF(N = 300))]
    twin, e, eb = run_twin(est)
    @printf("%-33s %6.3f %14.3f %16.4f ± %.4f\n", label, e, eb, twin[:α],
        uncertainty(twin, :α))
end
