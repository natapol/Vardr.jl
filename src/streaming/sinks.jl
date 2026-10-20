"""
$(TYPEDEF)

Receives the twin after every update made by [`run!`](@ref).

To add a sink, subtype `AbstractSink`, implement [`on_update`](@ref), and
optionally `Base.close(sink)`, which is called when the run ends.
"""
abstract type AbstractSink end

"""
    on_update(sink, twin, obs)

Called after the running twin assimilates `obs`, or after a clock tick (then `obs`
is `nothing`).
"""
function on_update end

Base.close(::AbstractSink) = nothing

"""
$(TYPEDEF)

Calls `f(twin, obs)` after every update.

$(TYPEDFIELDS)
"""
struct CallbackSink{F} <: AbstractSink
    f::F
end
on_update(s::CallbackSink, twin, obs) = s.f(twin, obs)

"""
$(TYPEDEF)

Writes a [`snapshot`](@ref) to `path` every `every` updates and when the run ends.

    SnapshotSink(path; every = 10)

$(TYPEDFIELDS)
"""
mutable struct SnapshotSink <: AbstractSink
    path::String
    every::Int
    count::Int
    twin::Any
end
SnapshotSink(path::AbstractString; every::Integer = 10) = SnapshotSink(String(path), every, 0,
    nothing)

function on_update(s::SnapshotSink, twin, obs)
    s.twin = twin
    s.count += 1
    s.count % s.every == 0 && snapshot(s.path, twin)
    return nothing
end

Base.close(s::SnapshotSink) = (s.twin === nothing || snapshot(s.path, s.twin); nothing)
