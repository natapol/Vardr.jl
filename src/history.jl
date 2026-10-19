"""
$(TYPEDEF)

Record of the twin's belief after every prediction and assimilation.

`History` is a Tables.jl column table with columns `t`, `kind` (`:init`, `:predict`
or `:update`), then `<name>` (posterior mean) and `<name>_std` for every augmented
state component. Turn it into a `DataFrame` with `DataFrame(history(twin))`.

$(TYPEDFIELDS)
"""
mutable struct History
    "Names of the augmented state components."
    names::Vector{Symbol}
    t::Vector{Float64}
    kind::Vector{Symbol}
    mean::Vector{Vector{Float64}}
    std::Vector{Vector{Float64}}
    "Observations that have been assimilated."
    observations::Vector{Observation}
end

History(names) = History(names, Float64[], Symbol[], Vector{Float64}[], Vector{Float64}[],
    Observation[])

function record!(h::History, t, kind, b::AbstractBelief)
    push!(h.t, t)
    push!(h.kind, kind)
    push!(h.mean, mean(b))
    push!(h.std, std(b))
    return h
end

Base.length(h::History) = length(h.t)

"Drop history records after the first `n` (observations are kept)."
function truncate!(h::History, n::Integer)
    foreach(v -> resize!(v, n), (h.t, h.kind, h.mean, h.std))
    return h
end

Tables.istable(::Type{History}) = true
Tables.columnaccess(::Type{History}) = true
function Tables.columns(h::History)
    cols = Pair{Symbol, AbstractVector}[:t => h.t, :kind => h.kind]
    for (i, n) in enumerate(h.names)
        push!(cols, n => [m[i] for m in h.mean])
        push!(cols, Symbol(n, :_std) => [s[i] for s in h.std])
    end
    return NamedTuple(cols)
end
