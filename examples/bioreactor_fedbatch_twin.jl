# A process twin of a fed-batch bioreactor producing a recombinant protein (biopharma).
#
#   julia --project=examples -t auto examples/bioreactor_fedbatch_twin.jl
#
# E. coli grows on glucose in a 5 L batch until the glucose runs out (~7 h), then an
# exponential glucose feed holds growth at a set rate. The feed cannot keep rising: the
# cells' oxygen uptake rate (OUR) grows with the feed, and once it exceeds the vessel's
# oxygen transfer capacity (OTR_max) the culture becomes oxygen limited (acetate
# overflow, lost product). At t = 20 h the process engineer must choose a feed cap
# F_cap for the rest of the run (harvest at 36 h): as high as possible for product,
# with at most a 5% chance of exceeding OTR_max.
#
# Model (mass balances, Monod growth, Luedeking–Piret product formation):
#
#   dX/dt = mu X - (F/V) X                     biomass (g/L)
#   dS/dt = -(mu/Yxs + ms m(S)) X + (F/V)(Sf - S)   glucose (g/L)
#   dP/dt = (alpha mu + qp) X - (F/V) P        product (g/L)
#   dV/dt = F                                  volume (L)
#   mu = mu_max S/(Ks + S),  m(S) = S/(Ks + S),  F = f_pump min(F0 e^{mu_set (t - t_feed)}, F_cap)
#
# Sensors (process analytical technology): capacitance probe (biomass) and off-gas
# analyser (OUR) and load cell (volume) every 30 min; at-line glucose and offline HPLC
# titer only every 4 h (NaN in between). OUR = kO (mu/Yxs + ms m(S)) X is a nonlinear
# function of the state and of estimated parameters.
#
# This batch's cells grow more slowly, yield less biomass and produce more than the
# process development values, and the feed pump delivers 8% less than its setpoint.
# The twin learns all four from the data and is used to choose F_cap. A twin that runs
# open loop on the development values cannot support the decision.
# Step-by-step explanation with theory: docs, "Case study: bioprocess twin".

using Vardr, Distributions, LinearAlgebra, Random, Statistics, Printf

# --- Model -------------------------------------------------------------------------------
const PNAMES = [:mu_max, :Ks, :Yxs, :ms, :alpha, :qp, :kO, :Sf,
    :F0, :mu_set, :t_feed, :F_cap, :f_pump]

feedrate(p, t) = t < p[11] ? 0.0 : p[13] * min(p[9] * exp(p[10] * (t - p[11])), p[12])
uptake(S, p) = (p[1] / p[3] + p[4]) * S / (p[2] + S)   # specific glucose uptake (g/g/h)

function bioreactor(u, p, t)
    X, S, P, V = u
    Sp = max(S, 0.0)
    mu = p[1] * Sp / (p[2] + Sp)
    F = feedrate(p, t)
    D = F / V                                          # dilution rate (1/h)
    return [mu * X - D * X,
            -uptake(Sp, p) * X + D * (p[8] - S),
            (p[5] * mu + p[6]) * X - D * P,
            F]
end
our(u, p) = p[7] * uptake(max(u[2], 0.0), p) * u[1]     # mmol O2 / L / h

# Process development (PD) values. The exponential feed is designed from them so that
# feed = uptake at mu = mu_set: F0 = (mu_set/Yxs + ms) X V / Sf with X = 10 g/L, V = 5 L.
pd = [0.6, 0.05, 0.5, 0.04, 0.01, 0.004, 12.0, 500.0,
    0.024, 0.1, 7.0, Inf, 1.0]
model = TwinModel(bioreactor, [0.2, 20.0, 0.0, 5.0], pd;
    states = [:X, :S, :P, :V], params = PNAMES, tstops = [7.0])

const OTR_MAX = 180.0     # mmol/L/h, oxygen transfer capacity of the vessel
const T_DECIDE = 20.0
const T_HARVEST = 36.0

# --- The real batch (unknown to the twin) ------------------------------------------------
batch = copy(pd)
batch[[1, 3, 6, 13]] = [0.52, 0.44, 0.0065, 0.92]   # mu_max, Yxs, qp, f_pump
rng = Xoshiro(3)
function run_plant(p, u, t0, t1; rng, dt = 0.5)
    traj = Dict{Float64, Vector{Float64}}()
    t = t0
    for tk in (t0 + dt):dt:t1
        pk = copy(p)
        pk[13] *= 1 + 0.03 * randn(rng)    # pump delivery fluctuates ±3%
        u = Vardr.propagate(model, u, pk, t, tk)
        t = tk
        traj[tk] = u
    end
    return traj
end
truth = run_plant(batch, [0.25, 20.0, 0.0, 5.0], 0.0, T_DECIDE; rng)

# Sensors: [X_cap, OUR, V_load, S_atline, P_hplc]; NaN = no sample at that time.
σ = [0.8, 4.0, 0.02, 0.1, 0.05]
data = map(0.5:0.5:T_DECIDE) do t
    u = truth[t]
    sampled = t % 4 == 0
    Observation(t, [u[1] + σ[1] * randn(rng), our(u, batch) + σ[2] * randn(rng),
        u[4] + σ[3] * randn(rng),
        sampled ? u[2] + σ[4] * randn(rng) : NaN,
        sampled ? u[3] + σ[5] * randn(rng) : NaN])
end

# --- The twin ----------------------------------------------------------------------------
sensors = ObservationModel((u, p, t) -> [u[1], our(u, p), u[4], u[2], u[3]], σ;
    names = [:X_cap, :OUR, :V_load, :S_atline, :P_hplc])
estimate = [:mu_max, :Yxs, :qp, :f_pump]
# Prior: PD values with batch-to-batch variability
prior = MvNormal([0.2, 20.0, 0.0, 5.0, 0.6, 0.5, 0.004, 1.0],
    Diagonal([0.05, 0.5, 0.001, 0.05, 0.1, 0.05, 0.002, 0.05] .^ 2))
make_twin(estimator) = Twin(model, sensors; name = "batch-2417", estimator, estimate, prior,
    process_noise = [0.05, 0.005, 0.005, 0.002, 0.002, 0.002, 0.0001, 0.002],
    bounds = (zeros(8), fill(Inf, 8)), rng = Xoshiro(1))
twin = make_twin(EnKF(N = 200, inflation = 1.02))

println("Learning this batch (true: mu_max 0.52, Yxs 0.44, qp 0.0065, f_pump 0.92)")
for obs in data
    step!(twin, obs)
    if obs.t in (4.0, 7.0, 12.0, 20.0)
        @printf("  t = %4.1f h  mu_max %.3f ± %.3f   Yxs %.3f ± %.3f   qp %.4f ± %.4f   f_pump %.3f ± %.3f\n",
            obs.t, (x for s in estimate for x in (twin[s], uncertainty(twin, s)))...)
    end
end
open_loop = make_twin(NoAssimilation(N = 200))
predict!(open_loop, T_DECIDE)

# --- Decision: the feed cap --------------------------------------------------------------
# For each candidate cap, forecast to harvest. Each forecast sample k comes from member k
# (nsamples = N), so OUR is computed with that member's own parameters.
function assess(tw, F_cap)
    fc = scenario(tw, SetParameter(T_DECIDE, :F_cap => F_cap), T_HARVEST;
        saveat = T_DECIDE:0.25:T_HARVEST, name = "F_cap = $F_cap")
    Z = members(tw)
    X, S, P, V = fc[:X], fc[:S], fc[:P], fc[:V]
    peak = map(axes(X, 2)) do k
        p = copy(pd)
        p[[1, 3, 6, 13]] = Z[5:8, k]
        maximum(our([X[j, k], S[j, k]], p) for j in axes(X, 1))
    end
    product = P[end, :] .* V[end, :]                   # g of protein at harvest
    return (risk = mean(peak .> OTR_MAX), product = mean(product),
        lo = quantile(product, 0.05), hi = quantile(product, 0.95))
end
function plant_outcome(F_cap)
    p = copy(batch)
    p[12] = F_cap
    tr = run_plant(p, truth[T_DECIDE], T_DECIDE, T_HARVEST; rng = Xoshiro(99))
    return maximum(our(u, p) for u in values(tr)), tr[T_HARVEST][3] * tr[T_HARVEST][4]
end

caps = [Inf, 0.25, 0.225, 0.2125, 0.20, 0.1875, 0.175, 0.15]
println("\nFeed cap (L/h)   twin: P(OUR > OTR_max)  product g [5–95%]    open loop: P(OUR > OTR_max)  product g    plant: peak OUR  product g")
for c in caps
    a, b = assess(twin, c), assess(open_loop, c)
    pk, prod = plant_outcome(c)
    @printf("  %-14s %10.2f        %5.1f [%4.1f, %4.1f]  %16.2f        %8.1f     %13.0f %9.1f\n",
        c == Inf ? "none (exp.)" : string(c), a.risk, a.product, a.lo, a.hi, b.risk, b.product, pk, prod)
end
best = maximum(c for c in caps if assess(twin, c).risk <= 0.05)
println("\nTwin's choice: F_cap = $best L/h. The plant then peaks at OUR = ",
    round(plant_outcome(best)[1]; digits = 1), " mmol/L/h (limit $OTR_MAX).")
