"""
$(TYPEDEF)

Keeps a running twin advancing through data outages. Whenever `interval` wall-clock
seconds pass with no observation, the twin predicts `dt` model time units ahead.
Set `dt` to roughly the model time that elapses in `interval` seconds of the real
system.

Ticks are provisional. Before the first tick of a quiet period the runner checkpoints
the belief. If an observation then arrives that the ticks have already moved past, the
twin rolls back to the checkpoint and assimilates it, so data is never dropped
because of a tick. The rolled-back tick records are removed from the history.

$(TYPEDFIELDS)
"""
struct WallClock
    interval::Float64
    dt::Float64
end
WallClock(; interval::Real, dt::Real) = WallClock(interval, dt)

"""
$(TYPEDEF)

Handle to a twin running on a data stream, returned by [`run!`](@ref). Use
`wait(runner)` to block until it finishes and [`stop!`](@ref) to end it.

$(TYPEDFIELDS)
"""
mutable struct TwinRunner{T <: Twin}
    twin::T
    task::Task
    inbox::Channel{Any}
    source_channel::Channel{Observation}
    owns_source::Bool
    "Observations assimilated."
    assimilated::Int
    "Observations dropped because they arrived after the twin had moved past them."
    late::Int
    "Clock ticks processed."
    ticks::Int
    "Times the twin was rolled back over provisional ticks to accept delayed data."
    rollbacks::Int
    "Belief, time and history length before the current run of ticks (or `nothing`)."
    checkpoint::Any
    "Observations that raised an error during assimilation."
    failed::Int
    "Wall-clock time (`time()`) at which the last observation arrived or tick ran."
    @atomic last_activity::Float64
    @atomic stopping::Bool
end

Base.show(io::IO, r::TwinRunner) = print(io, "TwinRunner(", r.twin.name, ", ",
    istaskdone(r.task) ? "finished" : "running", ", assimilated = ", r.assimilated,
    ", late = ", r.late, ", rollbacks = ", r.rollbacks, ")")

Base.wait(r::TwinRunner) = (wait(r.task); r)
Base.istaskdone(r::TwinRunner) = istaskdone(r.task)

"""
    run!(twin, source; sinks = [], until = Inf, clock = nothing, on_error = :stop, async = false)

Keep `twin` synchronized with a live [`AbstractDataSource`](@ref): every incoming
observation is assimilated with [`step!`](@ref), then passed to each sink.

- Observations older than the twin's current time (late or out of order) are
  dropped and counted in `runner.late`, unless only provisional clock ticks have
  moved the twin past them (see [`WallClock`](@ref)).
- `until`: stop once model time reaches this value.
- `clock`: a [`WallClock`](@ref) that advances the twin when no data arrives for a while.
- `on_error`: `:stop` ends the run and rethrows on `wait`; `:skip` logs a warning,
  counts the failure and continues.
- `async = true` returns the [`TwinRunner`](@ref) immediately. Otherwise `run!`
  blocks until the source is exhausted, `until` is reached, or [`stop!`](@ref) is
  called.

While a run is in progress, other tasks can still query the twin (`state(twin)`,
`forecast(twin, …)`), because every operation on the twin takes its lock.
"""
function run!(twin::Twin, source::AbstractDataSource; sinks = AbstractSink[],
        until::Real = Inf, clock::Union{WallClock, Nothing} = nothing,
        on_error::Symbol = :stop, async::Bool = false)
    on_error in (:stop, :skip) || throw(ArgumentError("on_error must be :stop or :skip"))
    sinks = collect(sinks)
    inbox = Channel{Any}(Inf)
    srcch = start(source)
    r = TwinRunner(twin, Task(() -> nothing), inbox, srcch, owns_channel(source), 0, 0, 0, 0, nothing, 0,
        time(), false)

    # Forward observations from the source into the inbox, then signal the end.
    errormonitor(@async begin
        try
            for obs in srcch
                @atomic r.last_activity = time()
                put!(inbox, obs)
            end
            put!(inbox, :eof)
        catch e
            e isa InvalidStateException || rethrow()
        end
    end)

    if clock !== nothing
        errormonitor(@async while isopen(inbox)
            sleep(clock.interval)
            idle(r, clock) || continue
            try
                put!(inbox, :tick)
            catch e
                e isa InvalidStateException || rethrow()
            end
        end)
    end

    r.task = @async try
        for msg in inbox
            @atomic(r.stopping) && break
            msg === :eof && break
            if msg === :tick
                idle(r, clock) || continue  # data arrived after the tick was queued
                if r.checkpoint === nothing
                    r.checkpoint = lock(twin.lock) do
                        (deepcopy(twin.belief), twin.t, length(twin.history))
                    end
                end
                predict!(twin, min(twin.t + clock.dt, until))
                r.ticks += 1
                @atomic r.last_activity = time()
                notify_sinks(sinks, twin, nothing)
            elseif msg isa Observation
                msg.t > until && break
                if msg.t < twin.t
                    if r.checkpoint !== nothing && msg.t >= r.checkpoint[2]
                        rollback!(twin, r.checkpoint)
                        r.rollbacks += 1
                    else
                        r.late += 1
                        continue
                    end
                end
                try
                    step!(twin, msg)
                    r.assimilated += 1
                    r.checkpoint = nothing
                catch e
                    on_error === :stop && rethrow()
                    r.failed += 1
                    @warn "Failed to assimilate observation at t = $(msg.t)" exception = e
                    continue
                end
                notify_sinks(sinks, twin, msg)
            end
            twin.t >= until && break
        end
    finally
        shutdown!(r)
        foreach(close, sinks)
    end

    async || wait(r)
    return r
end

# Undo provisional ticks: restore the belief and time saved before them.
function rollback!(twin::Twin, (belief, t, n))
    lock(twin.lock) do
        twin.belief = belief
        twin.t = t
        truncate!(twin.history, n)
    end
    return twin
end

# A tick may run only when the stream has been silent for a full interval and no data is
# waiting, so that a tick never moves the twin past observations already received.
function idle(r::TwinRunner, clock::WallClock)
    !isready(r.inbox) && !isready(r.source_channel) &&
        time() - @atomic(r.last_activity) >= clock.interval
end

notify_sinks(sinks, twin, obs) = foreach(s -> on_update(s, twin, obs), sinks)

function shutdown!(r::TwinRunner)
    close(r.inbox)
    r.owns_source && close(r.source_channel)
    return nothing
end

"""
    stop!(runner)

Stop a running twin after the observation currently being processed, and wait for
the run to finish.
"""
function stop!(r::TwinRunner)
    @atomic r.stopping = true
    shutdown!(r)
    try
        wait(r.task)
    catch
    end
    return r
end
