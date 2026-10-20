```@meta
CurrentModule = Vardr
```

# Running live

So far we have stepped twins through data in a loop. In operation, data arrives
asynchronously from sensors, message brokers or databases. [`run!`](@ref) connects a
twin to a **data source** and calls **sinks** after every update:

```
 source ──▶ run!(twin, source) ──▶ step!(twin, obs) ──▶ sinks (log, alert, save, publish)
              ▲         │
          WallClock     └─ other tasks may query the twin at any time
```

```@setup live
using Vardr, Distributions, LinearAlgebra, Random, Statistics
osc(u, p, t) = [u[2], -p[1] * u[1] - p[2] * u[2]]
real_system = TwinModel(osc, [1.5, 0.0], [3.0, 0.1])
truth(t) = Vardr.propagate(real_system, [1.5, 0.0], [3.0, 0.1], 0.0, t)
model = TwinModel(osc, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
sensors = ObservationModel(model, [:x], 0.05)
new_twin() = Twin(model, sensors; estimator = EnKF(N = 50), estimate = [:k],
    prior = MvNormal([1.0, 0.0, 4.0], Diagonal([0.5, 0.5, 1.0] .^ 2)),
    process_noise = 0.01, rng = Xoshiro(1))
```

## Replaying recorded data: `TableSource`

[`TableSource`](@ref) turns any Tables.jl table (a `DataFrame`, `CSV.File`, or named
tuple of vectors) into a stream. It is the easiest way to test a live setup against
historical data.

```@example live
rng = Xoshiro(7)
ts = 0.1:0.1:10.0
table = (time = collect(ts), x = [truth(t)[1] + 0.05 * randn(rng) for t in ts])

twin = new_twin()
runner = run!(twin, TableSource(table; time = :time, columns = [:x]))
```

- `columns` lists the measured columns, in the observation model's output order.
  `missing` values become `NaN`, meaning not observed.
- `speed`: model time units per wall-clock second. The default `Inf` replays as fast as
  possible; `speed = 60.0` replays one model minute per second, to rehearse real-time
  behaviour.
- Rows are sorted by time first.

Without `async = true`, `run!` blocks until the source is exhausted.

## Pushing data from your code: `ChannelSource`

[`ChannelSource`](@ref) is the bridge to any data feed: a serial port, an MQTT or
Kafka client callback, an HTTP handler, a database poller. Your code calls
`put!(src, Observation(t, y))`; the twin runs in a background task.

```@example live
src = ChannelSource()
twin = new_twin()
runner = run!(twin, src; async = true)          # returns immediately

feed = Threads.@spawn for t in 0.1:0.1:5.0       # e.g. your sensor callback
    put!(src, Observation(t, truth(t)[1] + 0.05 * randn()))
end
wait(feed)
sleep(0.5)                                       # let the twin catch up
(t = twin.t, x = twin[:x], k = twin[:k])         # query while it runs
```

```@example live
stop!(runner)     # or close(src) to end the stream gracefully
```

A sketch for MQTT (with Mosquitto.jl) looks like this; any client library follows the
same pattern:

```julia
client = Mosquitto.Client("broker.local")
subscribe(client, "plant/line3/levels")
@async while true
    msg = take!(get_messages_channel(client))
    d = JSON3.read(msg.payload)
    put!(src, Observation(d.t, [d.h1, something(d.h3, NaN)]))
end
```

!!! warning "Use model time, not arrival time"
    `Observation.t` must be the time the measurement was **taken**, in the model's time
    units: for example hours since commissioning, `(timestamp - t_ref) / 3600`. Pick a
    reference time `t_ref` and use it everywhere.

## Sinks: reacting to updates

After every update, `run!` calls each sink with the twin and the observation (`nothing`
after a clock tick). A [`CallbackSink`](@ref) wraps any function:

```@example live
logger = CallbackSink() do twin, obs
    obs !== nothing && obs.t % 2 == 0 &&
        println("t = $(twin.t): k = $(round(twin[:k]; digits = 3)) ± $(round(uncertainty(twin, :k); digits = 3))")
end

alerts = Tuple{Float64, Float64}[]
alarm = CallbackSink() do twin, obs
    obs === nothing && return
    fc = forecast(twin, twin.t + 2.0; nsamples = 50)
    p = mean(maximum(fc[:x]; dims = 1) .> 0.9)     # P(x > 0.9 within 2 time units)
    p > 0.5 && push!(alerts, (twin.t, p))          # or @warn, send an e-mail, …
end

twin = new_twin()
run!(twin, TableSource(table; time = :time); sinks = [logger, alarm])
length(alerts), first(alerts, 3)
```

Typical sinks:

- write the state to a time-series database or dashboard;
- publish the estimate back to the control system;
- raise alerts from forecasts (see `examples/thermal_room_twin.jl`);
- save snapshots with [`SnapshotSink`](@ref)`(path; every = 100)`, which also saves at
  shutdown.

Sinks run in the runner's task, so a slow sink delays the twin. Hand heavy work (long
forecasts, network calls) to another task, for example through a `Channel`.

### Writing your own sink

Subtype [`AbstractSink`](@ref), implement [`on_update`](@ref), and optionally
`Base.close`, which is called once when the run ends:

```@example live
mutable struct CSVSink <: AbstractSink
    io::IO
end
function Vardr.on_update(s::CSVSink, twin, obs)
    println(s.io, twin.t, ",", twin[:x], ",", uncertainty(twin, :x))
end
Base.close(s::CSVSink) = flush(s.io)

buf = IOBuffer()
run!(new_twin(), TableSource(table; time = :time); sinks = [CSVSink(buf)])
first(split(String(take!(buf)), '\n'), 3)
```

## Writing your own source

Subtype [`AbstractDataSource`](@ref) and implement `Vardr.start(src)`, which returns a
`Channel{Observation}`. If the source creates the channel itself, also define
`Vardr.owns_channel(src) = true`, so that [`stop!`](@ref) can close it.

```julia
struct PollingSource <: AbstractDataSource
    url::String
    every::Float64            # seconds
end
Vardr.owns_channel(::PollingSource) = true
function Vardr.start(src::PollingSource)
    Channel{Observation}(64) do ch
        while true
            r = JSON3.read(HTTP.get(src.url).body)
            put!(ch, Observation(r.t, [r.temperature, r.humidity]))
            sleep(src.every)
        end
    end
end
```

## Outages and the wall clock

If sensors go quiet, the twin would otherwise stop at the last observation time. A
[`WallClock`](@ref) keeps it moving: after `interval` seconds of silence, it predicts
`dt` model time units ahead and notifies the sinks with `obs = nothing`. The uncertainty
grows during the outage, honestly reflecting that the twin is "flying blind".

```julia
run!(twin, src; clock = WallClock(interval = 60.0, dt = 1/60), async = true)
#                                  1 min of silence → advance 1 min (time in hours)
```

Set `dt` to the model time that passes in `interval` seconds of real time.

Clock ticks are **provisional**. If a delayed observation then arrives with a time the
ticks have already passed, the twin rolls back to the state before the ticks and
assimilates it, so no data is lost. `runner.rollbacks` counts these events.

## Late and out-of-order data

A twin only moves forward in time. In a live run:

| Situation | What happens |
|:---|:---|
| Observation newer than the twin | predict to it, then assimilate (normal case) |
| Older, but only provisional ticks passed it | roll back, assimilate (`runner.rollbacks += 1`) |
| Older than the last assimilated observation | dropped (`runner.late += 1`) |
| Same time as the twin | assimilated (several sensors at one time) |

If your feed is often out of order, buffer it briefly and sort before `put!`.

## Errors, stopping, and the runner

```julia
runner = run!(twin, source; sinks, until = Inf, clock = nothing,
              on_error = :stop, async = false)
```

- `until`: stop once model time reaches this value.
- `on_error = :stop` (default): an exception while assimilating ends the run, and is
  rethrown by `wait(runner)`. `on_error = :skip` logs a warning, counts it in
  `runner.failed`, and continues. Use `:skip` in production, together with monitoring
  of `runner.failed`.
- [`stop!`](@ref)`(runner)`: finish the current observation, then stop. Sinks are
  closed.
- `wait(runner)`: block until the run ends. `istaskdone(runner)` checks without
  blocking.

The [`TwinRunner`](@ref) counts `assimilated`, `late`, `ticks`, `rollbacks` and
`failed`. Printing it gives a one-line status.

## Concurrency

Every operation on a twin (`step!`, `forecast`, `state`, `snapshot`, …) takes the twin's
lock, so a dashboard, REST endpoint or REPL can query a running twin safely. Forecasts
release the lock after drawing their samples, so a long what-if analysis does not block
assimilation.

Start Julia with several threads (`julia -t auto`). Ensemble members are integrated in
parallel, and a sensor thread, the runner and your queries can run side by side.

## Inputs that arrive live

Known inputs that change during operation, such as setpoints, valve positions or doses
given, are part of the model's forcing, not observations. Keep a time-stamped log that
the model reads, and append to it as changes arrive:

```julia
const valve_log = [(0.0, 0.5)]                         # (time, opening)
valve(t) = valve_log[findlast(e -> e[1] <= t, valve_log)][2]
tank(u, p, t) = [p[1] * valve(t) - p[2] * sqrt(max(u[1], 0))]

on_valve_change(t, opening) = lock(twin.lock) do       # from the control system
    push!(valve_log, (t, opening))
end
```

Taking the twin's lock while you append means no integration ever sees a half-updated
log. Record changes before (or with) the sensor data that follows them.

## A production checklist

1. Validate the twin on synthetic data, then replay historical data with
   `TableSource(...; speed = ...)`.
2. Use a fixed time reference, and model-time timestamps on every observation.
3. Use `on_error = :skip`, a `WallClock` if outages matter, and a `SnapshotSink`.
4. On restart, rebuild the twin from code, [`restore!`](@ref) the last snapshot, and
   resume the feed from `twin.t`.
5. Monitor `runner.late`, `runner.failed` and the twin's uncertainty. A twin whose
   uncertainty keeps growing despite data, or whose innovations are biased, has a model
   that no longer matches the system. That is often the most valuable alarm of all.
