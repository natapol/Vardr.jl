# A live digital twin of a heated room, running on a real-time sensor stream.
#
#   julia --project=examples -t auto examples/thermal_room_twin.jl
#
# Model (2R2C grey box, time in hours): room air and the building's thermal mass
#
#   Ca dTa/dt = (Tw - Ta)/Raw + (Tout(t) - Ta)/Rv + η P heat(Ta, t)
#   Cw dTw/dt = (Ta - Tw)/Raw + (Tout(t) - Tw)/Rwo
#
# A thermostat (known control law) drives the heater towards 21 °C by day and 16 °C at
# night. The weather forecast includes a cold snap on day 3.
#
# Two sensors stream in: room air temperature and an embedded wall-surface probe. The
# twin learns the envelope resistance Rwo (insulation quality) and the heater
# efficiency η while the data streams in. Every
# hour a sink forecasts the next 6 hours and raises an alert if the room is likely to
# drop below 19 °C while occupied, i.e. the heater cannot keep up. On day 2 the sensor
# goes offline for 6 hours. The twin keeps advancing on the wall clock until data
# returns.
#
# One simulated hour = 0.1 s of wall-clock time, so 3 days run in about 7 s.

using Vardr, Distributions, LinearAlgebra, Random, Statistics, Printf

# --- Known inputs: weather forecast and thermostat ---------------------------------------
# Daily cycle plus a cold snap of -14 °C arriving around t = 60 h
outdoor(t) = 2.0 + 5.0 * sin(2π * (t - 9) / 24) - 14.0 / (1 + exp(-(t - 60.0)))
occupied(t) = 6 <= mod(t, 24) < 22
setpoint(t) = 5 <= mod(t, 24) < 22 ? 21.0 : 16.0   # preheats from 05:00
heating(Ta, t) = clamp(2.0 * (setpoint(t) - Ta), 0.0, 1.0)   # proportional, saturating

# Parameters: Ca, Cw (kWh/K), Raw, Rv, Rwo (K/kW), P (kW), η
function room(u, p, t)
    Ta, Tw = u
    Ca, Cw, Raw, Rv, Rwo, P, η = p
    To = outdoor(t)
    [((Tw - Ta) / Raw + (To - Ta) / Rv + η * P * heating(Ta, t)) / Ca,
        ((Ta - Tw) / Raw + (To - Tw) / Rwo) / Cw]
end
pnames = [:Ca, :Cw, :Raw, :Rv, :Rwo, :P, :η]
# The setpoint switches at known hours; stop the solver there so it doesn't step over them.
switches = sort(vcat([[24d + 5, 24d + 22] for d in 0:4]...))
building(p, u0) = TwinModel(room, u0, p; states = [:Ta, :Tw], params = pnames,
    tstops = float.(switches))

# --- The real building: worse insulation and a less efficient heater than designed -------
real_p = [0.5, 15.0, 2.0, 25.0, 4.0, 8.0, 0.8]
real_building = building(real_p, [19.0, 17.0])

# --- The twin, built from design values ---------------------------------------------------
design_p = [0.5, 15.0, 2.0, 25.0, 7.0, 8.0, 1.0]
model = building(design_p, [19.0, 17.0])
sensors = ObservationModel(model, [:Ta, :Tw], [0.2, 0.3])
prior = MvNormal([19.0, 17.0, 7.0, 1.0], Diagonal([0.5, 1.5, 2.5, 0.2] .^ 2))
twin = Twin(model, sensors; name = "room-101", estimator = EnKF(N = 60, inflation = 1.01),
    estimate = [:Rwo, :η], prior, rng = Xoshiro(3),
    process_noise = [0.05, 0.02, 0.01, 0.002], bounds = ([-Inf, -Inf, 0.5, 0.1], fill(Inf, 4)))

# --- Live operation ----------------------------------------------------------------------
const SPEED = 10.0           # simulated hours per wall-clock second
const OUTAGE = (30.0, 36.0)  # sensor offline (hours), at night
src = ChannelSource()

# The "sensor": simulates the building in real time and pushes a reading every 15 min.
sensor = Threads.@spawn begin
    rng = Xoshiro(99)
    u, t = [19.0, 17.0], 0.0
    while t < 72.0
        sleep(0.25 / SPEED)
        u = Vardr.propagate(real_building, u, real_p, t, t + 0.25)
        t += 0.25
        OUTAGE[1] <= t < OUTAGE[2] && continue  # offline: nothing is sent
        put!(src, Observation(t, u .+ [0.2, 0.3] .* randn(rng, 2)))
    end
    close(src)
end

# Sinks: a log line every 6 h, and an hourly comfort check based on a 6 h forecast
log_sink = CallbackSink() do tw, obs
    tw.t % 6 == 0 || return
    @printf("%5.1f h  Ta = %5.2f ± %.2f °C   Rwo = %4.2f ± %.2f   η = %4.2f ± %.2f %s\n",
        tw.t, tw[:Ta], uncertainty(tw, :Ta), tw[:Rwo], uncertainty(tw, :Rwo), tw[:η],
        uncertainty(tw, :η), obs === nothing ? "  (no sensor: clock tick)" : "")
end
alerts = Float64[]
comfort_sink = CallbackSink() do tw, obs
    (obs === nothing || obs.t % 1 != 0) && return
    fc = forecast(tw, tw.t + 6; saveat = tw.t:0.5:(tw.t + 6), nsamples = 60)
    occ = occupied.(fc.t)
    any(occ) || return
    p_cold = mean(minimum(fc[:Ta][occ, :]; dims = 1) .< 19.0)
    if p_cold > 0.5 && (isempty(alerts) || tw.t - alerts[end] >= 6)
        push!(alerts, tw.t)
        @printf("%5.1f h  ⚠ comfort alert: %d%% chance the occupied room drops below 19 °C within 6 h\n",
            tw.t, round(Int, 100p_cold))
    end
end
snapshot_path = joinpath(mktempdir(), "room-101.jld2")

runner = run!(twin, src; async = true,
    sinks = [log_sink, comfort_sink, SnapshotSink(snapshot_path; every = 48)],
    # After 0.1 s (= 1 simulated hour) of silence, advance 1 h. Ticks are provisional:
    # if a delayed reading turns up, the twin rolls back and assimilates it.
    clock = WallClock(interval = 1.0 / SPEED, dt = 1.0))

# The twin can be queried from other tasks while it runs (e.g. a dashboard or API).
sleep(3.2)
@printf("\n[dashboard] t = %.1f h, room at %.2f °C (sensor %s)\n\n", twin.t, twin[:Ta],
    OUTAGE[1] <= twin.t < OUTAGE[2] ? "OFFLINE" : "online")
wait(runner)

println("\n", runner)
@printf("Learned  Rwo = %.2f ± %.2f K/kW (design 7.0, real 4.0)\n", twin[:Rwo],
    uncertainty(twin, :Rwo))
@printf("Learned  η   = %.2f ± %.2f      (design 1.0, real 0.8)\n", twin[:η],
    uncertainty(twin, :η))

# Uncertainty during the outage, from the history
h = history(twin)
before = findlast(i -> h.kind[i] == :update && h.t[i] < OUTAGE[1], eachindex(h.t))
during = findall(t -> OUTAGE[1] < t <= OUTAGE[2], h.t)
@printf("Tw uncertainty: %.2f °C before the outage, up to %.2f °C during it (the thermostat keeps Ta near its setpoint)\n",
    h.std[before][2], maximum(h.std[i][2] for i in during))

# What-if for the building manager: during the cold snap, would an insulation retrofit
# (Rwo → 8) or a bigger heater (P → 12 kW) keep the room comfortable tomorrow?
t0 = twin.t
saveat = t0:0.25:(t0 + 24)
options = [forecast(twin, t0 + 24; saveat, name = "as is"),
    scenario(twin, SetParameter(t0, :Rwo => 8.0), t0 + 24; saveat, name = "insulation retrofit"),
    scenario(twin, SetParameter(t0, :P => 12.0), t0 + 24; saveat, name = "12 kW heater")]
println("\nTomorrow (cold snap)     occupied hours below 19 °C   coldest occupied °C")
for fc in options
    occ = occupied.(fc.t)
    Ta = mean(fc)[1, occ]
    @printf("%-24s %16.1f h %24.1f\n", fc.name, 0.25 * count(<(19.0), Ta), minimum(Ta))
end
println("Final snapshot written to ", snapshot_path)
