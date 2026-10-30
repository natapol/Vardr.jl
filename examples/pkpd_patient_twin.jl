# A personalized pharmacokinetic/pharmacodynamic (PK/PD) twin of one patient.
#
#   julia --project=examples -t auto examples/pkpd_patient_twin.jl
#
# Model: two-compartment PK with repeated IV bolus doses, plus an effect compartment
# driving an Emax response:
#
#   dAc/dt = -(CL/V) Ac - (Q/V) Ac + (Q/Vp) Ap      central amount (mg)
#   dAp/dt =  (Q/V) Ac - (Q/Vp) Ap                  peripheral amount (mg)
#   dCe/dt =  ke0 (Ac/V - Ce)                       effect-site concentration (mg/L)
#   E      =  Emax Ce / (EC50 + Ce)                 response (e.g. % inhibition)
#
# The twin starts from population-typical parameters. It sees a few sparse plasma
# concentration samples and a frequent biomarker of the response, learns this
# patient's clearance (CL) and sensitivity (EC50), and is then used to compare dosing
# regimens before they are given.

using Vardr, Distributions, LinearAlgebra, Random, Statistics
using SciMLBase: DiscreteCallback

const DOSE_INTERVAL = 12.0                       # hours
const DOSE_TIMES = collect(DOSE_INTERVAL:DOSE_INTERVAL:240.0)  # first dose is in u0
const PNAMES = [:CL, :V, :Q, :Vp, :ke0, :Emax, :EC50, :dose]

function pkpd(u, p, t)
    Ac, Ap, Ce = u
    CL, V, Q, Vp, ke0 = p[1], p[2], p[3], p[4], p[5]
    [-(CL / V) * Ac - (Q / V) * Ac + (Q / Vp) * Ap,
        (Q / V) * Ac - (Q / Vp) * Ap,
        ke0 * (Ac / V - Ce)]
end

# Repeated bolus: at every dose time add `dose` (parameter 8) to the central compartment.
# Because the amount is a parameter, a what-if scenario can change it with SetParameter.
dosing = DiscreteCallback((u, t, integ) -> t in DOSE_TIMES,
    integ -> (integ.u[1] += integ.p[8]); save_positions = (true, true))
pk_model(p, dose0) = ODEModel(pkpd, [dose0, 0.0, 0.0], p; states = [:Ac, :Ap, :Ce],
    params = PNAMES, callback = dosing, tstops = DOSE_TIMES)

concentration(u, p) = u[1] / p[2]
response(u, p) = p[6] * u[3] / (p[7] + u[3])

# --- The patient (unknown to the twin): clears the drug fast and is quite sensitive ------
patient_p = [7.5, 40.0, 8.0, 60.0, 0.4, 100.0, 1.2, 100.0]
patient = pk_model(patient_p, 100.0)

# Sparse plasma samples (proportional ~10% error) and a biomarker every 2 h (±3 units).
# Each row is [concentration, response]. NaN = not measured at that time.
rng = Xoshiro(2024)
sample_times = sort(unique(vcat(2.0:2.0:96.0, [0.5, 6.0, 11.9, 36.5, 47.9, 72.5, 83.9])))
truth_at(t) = Vardr.propagate(patient, [100.0, 0.0, 0.0], patient_p, 0.0, t)
data = map(sample_times) do t
    u = truth_at(t)
    c = t in (0.5, 6.0, 11.9, 36.5, 47.9, 72.5, 83.9) ?
        concentration(u, patient_p) * (1 + 0.1 * randn(rng)) : NaN
    Observation(t, [c, response(u, patient_p) + 3.0 * randn(rng)])
end

# --- The twin: population-typical prior, joint estimation of CL and EC50 -----------------
population_p = [5.0, 40.0, 8.0, 60.0, 0.4, 100.0, 2.5, 100.0]
model = pk_model(population_p, 100.0)
sensors = ObservationModel((u, p, t) -> [concentration(u, p), response(u, p)],
    [0.25, 3.0]; names = [:C, :E])
# Prior: states known (dose just given); CL and EC50 vary ~40% between patients
prior = MvNormal([100.0, 0.0, 0.0, 5.0, 2.5], Diagonal([1e-6, 1e-6, 1e-6, 2.0, 1.0] .^ 2))
twin = Twin(model, sensors; name = "patient-042", estimator = EnKF(N = 200),
    estimate = [:CL, :EC50], prior, rng = Xoshiro(7),
    process_noise = [0.0, 0.0, 0.0, 0.002, 0.002],
    bounds = (zeros(5), fill(Inf, 5)))

println("time   CL estimate        EC50 estimate     (patient: CL = 7.5, EC50 = 1.2)")
for obs in data
    step!(twin, obs)
    if obs.t in (6.0, 24.0, 48.0, 96.0)
        println(lpad(obs.t, 4), " h   ", rpad("$(round(twin[:CL]; digits = 2)) ± $(round(uncertainty(twin, :CL); digits = 2))", 19),
            round(twin[:EC50]; digits = 2), " ± ", round(uncertainty(twin, :EC50); digits = 2))
    end
end

# --- What-if: compare regimens over the next 48 h before giving them ---------------------
# Targets: response above 50% (efficacy) and peak concentration below 6 mg/L (toxicity).
# The forecast ends just before a dose time, so its last point is a true trough.
t_now = twin.t
horizon = t_now + 47.75
saveat = t_now:0.25:horizon
regimens = [
    forecast(twin, horizon; saveat, name = "continue 100 mg q12h"),
    scenario(twin, SetParameter(t_now + 1, :dose => 150.0), horizon; saveat,
        name = "increase to 150 mg q12h"),
    scenario(twin, SetParameter(t_now + 1, :dose => 200.0), horizon; saveat,
        name = "increase to 200 mg q12h"),
    scenario(twin, [SetParameter(t_now + 11, :dose => 0.0),     # dose at 108 h is missed
            SetParameter(t_now + 13, :dose => 100.0)], horizon; saveat,
        name = "missed dose at 108 h")]

# Recompute concentration and response for each sample, using that sample's parameters
# (the forecast does not store parameters, so use the posterior means of CL and EC50).
p_hat = copy(population_p)
p_hat[1], p_hat[7] = twin[:CL], twin[:EC50]
println("\nRegimen (next 48 h)            time E>50%   P(Cmax > 6 mg/L)   trough C (mg/L)")
for fc in regimens
    Ac, Ce = fc[:Ac], fc[:Ce]
    E = p_hat[6] .* Ce ./ (p_hat[7] .+ Ce)
    C = Ac ./ p_hat[2]
    frac_eff = mean(E .> 50)
    p_tox = mean(maximum(C; dims = 1) .> 6)
    trough = median(C[end, :])
    println(rpad(fc.name, 31), lpad("$(round(Int, 100frac_eff))%", 11),
        lpad("$(round(Int, 100p_tox))%", 19), lpad(round(trough; digits = 2), 18))
end
