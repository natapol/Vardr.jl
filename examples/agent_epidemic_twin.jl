# An agent-based twin of an epidemic in a town.
#
#   julia --project=examples -t auto examples/agent_epidemic_twin.jl
#
# 5000 people in three age groups mix through a contact matrix; each infection lasts
# exactly 5 days (not the exponential duration an ODE model implies), and a fraction of
# infected elders are admitted to hospital. The twin follows reported cases and hospital
# admissions with a particle filter, learns the transmission rate and the reporting
# fraction, and compares a school closure with shielding the elderly. Both
# interventions act on individual agents, which an aggregate ODE model cannot express
# directly.

using Vardr, Random, Statistics, LinearAlgebra, Distributions, Printf

const GROUPS = 3                          # children, adults, elders
const CONTACTS = [8.0 3.0 1.0             # daily contacts of a group-g person
                  3.0 6.0 1.5             # with group h (rows g, columns h)
                  1.0 1.5 3.0]
const INFECTIOUS_DAYS = 5
const P_HOSPITAL = 0.08                   # admissions per infected elder

mutable struct Town
    group::Vector{Int8}
    status::Vector{Int8}                  # 0 susceptible, 1 infectious, 2 recovered
    days_left::Vector{Int8}
    contacts::Matrix{Float64}             # can be changed by interventions
    new_cases::Int
    new_admissions::Int
end

# p = [β, ρ]: transmission probability per contact, reporting fraction
function town(u, p, rng)
    n = 5000
    group = Int8.(rand(rng, Categorical([0.25, 0.55, 0.20]), n))
    status = zeros(Int8, n)
    days_left = zeros(Int8, n)
    for i in randperm(rng, n)[1:clamp(round(Int, u[2]), 1, n)]
        status[i], days_left[i] = 1, rand(rng, 1:INFECTIOUS_DAYS)
    end
    Town(group, status, days_left, copy(CONTACTS), 0, 0)
end

function town_step!(town, p, t, rng)
    β = p[1]
    size_g = [count(==(g), town.group) for g in 1:GROUPS]
    infectious_g = zeros(GROUPS)
    for i in eachindex(town.status)
        town.status[i] == 1 && (infectious_g[town.group[i]] += 1)
    end
    # Probability that a group-g susceptible is infected today
    p_inf = [1 - exp(-β * sum(town.contacts[g, h] * infectious_g[h] / size_g[h]
                              for h in 1:GROUPS)) for g in 1:GROUPS]
    town.new_cases = town.new_admissions = 0
    for i in eachindex(town.status)
        if town.status[i] == 1
            town.days_left[i] -= 1
            town.days_left[i] == 0 && (town.status[i] = 2)
        elseif town.status[i] == 0 && rand(rng) < p_inf[town.group[i]]
            town.status[i], town.days_left[i] = 1, INFECTIOUS_DAYS
            town.new_cases += 1
            town.group[i] == 3 && rand(rng) < P_HOSPITAL && (town.new_admissions += 1)
        end
    end
    return town
end

town_summary(town) = [count(==(0), town.status), count(==(1), town.status),
    count(==(2), town.status), town.new_cases, town.new_admissions]

model = AgentModel(town, town_step!, town_summary, [4980, 20, 0, 0, 0], [0.025, 0.4];
    states = [:S, :I, :R, :cases, :admissions], params = [:β, :ρ])
println(model)

# --- The real town: β = 0.03, a third of cases are reported ---------------------------
p_true = [0.03, 0.33]
real_town = town([4980, 20, 0, 0, 0], p_true, Xoshiro(1))
rng = Xoshiro(2)
days = 1:25
true_infectious = Int[]
data = map(days) do d
    town_step!(real_town, p_true, d - 1, rng)
    push!(true_infectious, count(==(1), real_town.status))
    Observation(d, [p_true[2] * real_town.new_cases + 3 * randn(rng),
        real_town.new_admissions + 0.5 * randn(rng)])
end

# Daily reported cases (ρ × new infections) and hospital admissions
surveillance = ObservationModel((u, p, t) -> [p[2] * u[4], u[5]], [3.0, 0.5];
    names = [:reported, :admissions])

prior = MvNormal([4980, 20, 0, 0, 0, 0.025, 0.4], Diagonal([1, 5, 1, 1, 1, 0.01, 0.15] .^ 2))
twin = Twin(model, surveillance; estimator = ParticleFilter(N = 400, jitter = 0.2),
    estimate = [:β, :ρ], prior, process_noise = [0, 0, 0, 0, 0, 0.001, 0.005],
    bounds = ([-Inf, -Inf, -Inf, -Inf, -Inf, 0.0, 0.05], [Inf, Inf, Inf, Inf, Inf, Inf, 1.0]),
    rng = Xoshiro(3))

for o in data
    step!(twin, o)
    o.t % 5 == 0 && @printf("day %2d: β = %.4f ± %.4f, ρ = %.2f ± %.2f, infectious ≈ %4.0f (true %d)\n",
        o.t, twin[:β], uncertainty(twin, :β), twin[:ρ], uncertainty(twin, :ρ), twin[:I],
        true_infectious[Int(o.t)])
end

# --- What if? -------------------------------------------------------------------------
close_schools = Event(27.0, (town, p, t) -> (town.contacts[1, 1] *= 0.2))
shield_elders = Event(27.0, (town, p, t) -> (town.contacts[3, :] .*= 0.3;
                                             town.contacts[:, 3] .*= 0.3))
horizon = 150.0
saveat = 26:150
options = [("no action", forecast(twin, horizon; saveat)),
    ("close schools on day 27", scenario(twin, close_schools, horizon; saveat)),
    ("shield elders on day 27", scenario(twin, shield_elders, horizon; saveat))]

println("\nDays 26–150              admissions (90% range)   people infected (90% range)")
for (name, fc) in options
    admissions = vec(sum(fc[:admissions]; dims = 1))
    infected = 5000 .- fc[:S][end, :]
    @printf("  %-24s %5.0f (%3.0f–%3.0f)        %5.0f (%4.0f–%4.0f)\n", name, mean(admissions),
        quantile(admissions, 0.05), quantile(admissions, 0.95), mean(infected),
        quantile(infected, 0.05), quantile(infected, 0.95))
end
