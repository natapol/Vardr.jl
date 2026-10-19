"""
$(TYPEDEF)

A stream of [`Observation`](@ref)s feeding a running twin (see [`run!`](@ref)).

To add a source, subtype `AbstractDataSource` and implement
`Vardr.start(src)::Channel{Observation}`. If the source creates that channel
itself, also define `Vardr.owns_channel(src) = true` so the runner can close it on
[`stop!`](@ref).
"""
abstract type AbstractDataSource end

function start end
owns_channel(::AbstractDataSource) = false

"""
$(TYPEDEF)

Observations pushed by user code: `put!(src.channel, Observation(t, y))`.

    ChannelSource(; size = 1024)
    ChannelSource(channel::Channel{Observation})

$(TYPEDFIELDS)
"""
struct ChannelSource <: AbstractDataSource
    channel::Channel{Observation}
end
ChannelSource(; size::Integer = 1024) = ChannelSource(Channel{Observation}(size))
start(src::ChannelSource) = src.channel
Base.put!(src::ChannelSource, obs::Observation) = put!(src.channel, obs)
Base.close(src::ChannelSource) = close(src.channel)

"""
$(TYPEDEF)

Replays a table (anything Tables.jl-compatible, such as a `DataFrame` or
`CSV.File`) as a stream of observations, sorted by time.

    TableSource(table; time = :t, columns = nothing, speed = Inf)

- `columns`: the measured columns, in the order of the observation model's outputs
  (default: every column except `time`). `missing` becomes `NaN`, i.e. unobserved.
- `speed`: model time units per wall-clock second. `Inf` replays as fast as
  possible; a finite value paces the replay to simulate a live feed.

$(TYPEDFIELDS)
"""
struct TableSource <: AbstractDataSource
    observations::Vector{Observation}
    speed::Float64
end

function TableSource(table; time::Symbol = :t, columns = nothing, speed::Real = Inf)
    speed > 0 || throw(ArgumentError("speed must be positive"))
    cols = Tables.columns(table)
    names = Tables.columnnames(cols)
    time in names || throw(ArgumentError("table has no time column `$time`"))
    columns = columns === nothing ? [n for n in names if n != time] : collect(Symbol, columns)
    ts = Tables.getcolumn(cols, time)
    data = [Tables.getcolumn(cols, c) for c in columns]
    obs = [Observation(ts[i], Float64[_num(d[i]) for d in data]) for i in eachindex(ts)]
    TableSource(sort!(obs; by = o -> o.t), Float64(speed))
end

_num(::Missing) = NaN
_num(x::Real) = Float64(x)

owns_channel(::TableSource) = true

function start(src::TableSource)
    Channel{Observation}(32) do ch
        prev = isempty(src.observations) ? 0.0 : first(src.observations).t
        for o in src.observations
            if isfinite(src.speed) && o.t > prev
                sleep((o.t - prev) / src.speed)
            end
            prev = o.t
            put!(ch, o)
        end
    end
end
