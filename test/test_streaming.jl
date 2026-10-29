obs = synthetic_observations()
table = (t = [o.t for o in obs], x = [o.y[1] for o in obs])

@testset "table replay matches manual stepping" begin
    manual = oscillator_twin()
    foreach(o -> step!(manual, o), obs)

    live = oscillator_twin()
    seen = Ref(0)
    r = run!(live, TableSource(table); sinks = [CallbackSink((tw, o) -> seen[] += 1)])
    @test istaskdone(r)
    @test r.assimilated == length(obs)
    @test seen[] == length(obs)
    @test members(live) == members(manual)
end

@testset "until, missing values and paced replay" begin
    tw = oscillator_twin()
    tbl = (time = [0.1, 0.2, 0.3, 5.0], x = [1.0, missing, 0.9, 0.0])
    src = TableSource(tbl; time = :time, speed = 10.0)
    @test isnan(src.observations[2].y[1])
    r = run!(tw, src; until = 1.0)
    @test r.assimilated == 3
    @test tw.t == 0.3
end

@testset "late data and async stop" begin
    tw = oscillator_twin()
    src = ChannelSource()
    r = run!(tw, src; async = true)
    put!(src, Observation(1.0, [0.5]))
    put!(src, Observation(0.5, [0.5]))  # older than the twin: dropped
    put!(src, Observation(1.5, [0.2]))
    timedwait(() -> r.assimilated + r.late == 3, 30.0)
    @test r.assimilated == 2
    @test r.late == 1
    @test !istaskdone(r)
    stop!(r)
    @test istaskdone(r)
end

@testset "wall clock advances the twin without data" begin
    tw = oscillator_twin()
    src = ChannelSource()
    r = run!(tw, src; clock = WallClock(interval = 0.01, dt = 0.25), until = 1.0)
    @test tw.t == 1.0
    @test r.ticks == 4
    @test r.assimilated == 0
end

@testset "delayed data rolls back provisional ticks" begin
    tw = oscillator_twin()
    src = ChannelSource()
    r = run!(tw, src; clock = WallClock(interval = 0.01, dt = 1.0), async = true)
    timedwait(() -> r.ticks >= 2, 30.0)
    @test tw.t >= 2.0
    n0 = count(==(:init), history(tw).kind)
    put!(src, Observation(0.5, [1.0]))  # older than the ticked clock, newer than the last data
    timedwait(() -> r.assimilated == 1, 30.0)
    stop!(r)
    @test r.rollbacks == 1
    @test r.late == 0
    @test history(tw).observations[1].t == 0.5
    # the ticks before the delayed observation were removed from the history
    i = findfirst(==(:update), history(tw).kind)
    @test all(<=(0.5), history(tw).t[1:i])
    @test count(==(:init), history(tw).kind) == n0

    # data older than the last assimilated observation is still dropped
    tw = oscillator_twin()
    src = ChannelSource()
    r = run!(tw, src; clock = WallClock(interval = 0.01, dt = 1.0), async = true)
    put!(src, Observation(1.0, [0.5]))
    timedwait(() -> r.assimilated == 1, 30.0)
    put!(src, Observation(0.5, [0.5]))
    timedwait(() -> r.late == 1, 30.0)
    stop!(r)
    @test r.late == 1
end

@testset "error handling" begin
    bad(u, p, t) = t > 0.5 ? [NaN, NaN] : osc(u, p, t)
    m = ODEModel(bad, [1.0, 0.0], [4.0, 0.1]; states = [:x, :v], params = [:k, :c])
    tbl = (t = [0.2, 1.0], x = [0.9, 0.0])
    tw = Twin(m, ObservationModel(m, [:x], 0.1); estimator = EnKF(N = 10))
    @test_throws Exception run!(tw, TableSource(tbl))

    tw = Twin(m, ObservationModel(m, [:x], 0.1); estimator = EnKF(N = 10))
    r = @test_logs (:warn,) match_mode = :any run!(tw, TableSource(tbl); on_error = :skip)
    @test r.assimilated == 1
    @test r.failed == 1
end

@testset "snapshot sink" begin
    path = joinpath(mktempdir(), "live.jld2")
    tw = oscillator_twin()
    run!(tw, TableSource(table); sinks = [SnapshotSink(path; every = 25)])
    restored = restore!(oscillator_twin(), path)
    @test restored.t == tw.t
    @test members(restored) == members(tw)
end
