const SNAPSHOT_VERSION = 2

"""
    snapshot(path, twin)

Save the twin's dynamic state (time, belief, fixed parameters, history and RNG
state) to a JLD2 file. Model code and the observation function are not saved. The
belief includes the members' latent objects, such as the agents of an
[`AgentModel`](@ref), so these must be types that JLD2 can store.
Reconstruct the twin with the same code, then call [`restore!`](@ref).
"""
function snapshot(path::AbstractString, twin::Twin)
    lock(twin.lock) do
        JLD2.jldsave(path; version = SNAPSHOT_VERSION, name = twin.name, t = twin.t,
            belief = twin.belief, p = twin.p, history = twin.history,
            rng = copy(twin.rng), nz = size(twin.belief.X, 1))
    end
    return path
end

"""
    restore!(twin, path)

Load a [`snapshot`](@ref) into a twin built from the same model, observation model
and estimator. The twin continues exactly where the snapshot left off.
"""
function restore!(twin::Twin, path::AbstractString)
    d = JLD2.load(path)
    d["version"] == SNAPSHOT_VERSION ||
        throw(ArgumentError("unsupported snapshot version $(d["version"])"))
    d["nz"] == size(twin.belief.X, 1) ||
        throw(DimensionMismatch("snapshot has $(d["nz"]) state components; this twin has $(size(twin.belief.X, 1))"))
    d["belief"] isa typeof(twin.belief) ||
        throw(ArgumentError("snapshot belief is a $(typeof(d["belief"])); this twin uses $(typeof(twin.belief))"))
    lock(twin.lock) do
        twin.name = d["name"]
        twin.t = d["t"]
        twin.belief = d["belief"]
        twin.p = d["p"]
        twin.history = d["history"]
        twin.rng isa typeof(d["rng"]) && copy!(twin.rng, d["rng"])
    end
    return twin
end
