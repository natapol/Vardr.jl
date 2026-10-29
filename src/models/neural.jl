"""
$(TYPEDEF)

A small fully connected neural network (multilayer perceptron) whose weights are a
flat vector, so they can be estimated like any other parameter.

    MLP(sizes...; activation = tanh)
    net(x, w)          # evaluate with weight vector w

`MLP(2, 16, 16, 1)` maps 2 inputs through two hidden layers of 16 units to 1 output.
`activation` is applied after every layer except the last (which is linear).
[`nweights`](@ref) gives the length of `w`, and [`init_weights`](@ref) a random
starting point. The network is plain Julia, so it works with dual numbers
(gradient-based calibration) and needs no deep-learning package.

$(TYPEDFIELDS)
"""
struct MLP{F}
    "Layer widths, from inputs to outputs."
    sizes::Vector{Int}
    activation::F
end

function MLP(sizes::Integer...; activation = tanh)
    length(sizes) ≥ 2 || throw(ArgumentError("an MLP needs at least an input and an output size"))
    all(>(0), sizes) || throw(ArgumentError("layer sizes must be positive"))
    MLP(collect(Int, sizes), activation)
end

Base.show(io::IO, n::MLP) = print(io, "MLP(", join(n.sizes, " → "), ", ",
    nameof(n.activation), ", ", nweights(n), " weights)")

"""
    nweights(network)

Number of weights (including biases) of a network.
"""
nweights(n::MLP) = sum((n.sizes[i] + 1) * n.sizes[i + 1] for i in 1:(length(n.sizes) - 1))

function (n::MLP)(x, w)
    length(w) == nweights(n) ||
        throw(DimensionMismatch("got $(length(w)) weights; the network has $(nweights(n))"))
    h = x
    k = 0
    L = length(n.sizes) - 1
    for i in 1:L
        nin, nout = n.sizes[i], n.sizes[i + 1]
        W = reshape(view(w, (k + 1):(k + nin * nout)), nout, nin)
        k += nin * nout
        b = view(w, (k + 1):(k + nout))
        k += nout
        h = W * h .+ b
        i < L && (h = n.activation.(h))
    end
    return h
end

"""
    init_weights([rng], network; output_scale = 0.1)

Random starting weights: Glorot-uniform weights and zero biases, with the last layer
scaled down by `output_scale` so that the untrained network's output is small. In a
universal differential equation the model then starts close to its known part.
"""
function init_weights(rng::AbstractRNG, n::MLP; output_scale::Real = 0.1)
    w = Float64[]
    L = length(n.sizes) - 1
    for i in 1:L
        nin, nout = n.sizes[i], n.sizes[i + 1]
        a = sqrt(6 / (nin + nout)) * (i == L ? output_scale : 1)
        append!(w, a .* (2 .* rand(rng, nout * nin) .- 1))
        append!(w, zeros(nout))
    end
    return w
end
init_weights(n::MLP; kwargs...) = init_weights(Random.default_rng(), n; kwargs...)

"""
$(TYPEDEF)

An ODE whose right-hand side is, or contains, a neural network.

    NeuralODEModel(network, u0; f = nothing, p = Float64[], weights = nothing,
        states = nothing, params = nothing, rng = Random.default_rng(), t0 = 0.0,
        alg = Tsit5(), solve_kwargs...)

- **Neural ODE** (`f = nothing`): `du/dt = network(u, w)`. The network learns all of
  the dynamics.
- **Universal differential equation** (hybrid model): `du/dt = f(u, p, t, nn)`, where
  `nn(x) = network(x, w)`. Write the known physics in `f` and call `nn` for the
  part that is unknown, for example a reaction rate.

`network(x, w)` is any callable that takes an input vector and a flat weight vector:
an [`MLP`](@ref), or a model from a deep-learning package wrapped in a closure.
`weights` are the starting weights; by default they come from
[`init_weights`](@ref)`(rng, network)`, which [`MLP`](@ref) supports.

The weights are parameters named `nn1, nn2, …` (see [`weightnames`](@ref)), after the
mechanistic parameters `p`. Train them with [`calibrate!`](@ref) (offline,
preferably gradient-based), or estimate them online like any other parameter
(`estimate = weightnames(model)`).

$(TYPEDFIELDS)
"""
struct NeuralODEModel{N, M <: ODEModel} <: AbstractTwinModel
    network::N
    "The equivalent ODE model, whose parameters are `[p; w]`."
    ode::M
    "Number of mechanistic parameters (before the weights)."
    nparams::Int
end

function NeuralODEModel(network, u0; f = nothing, p = Float64[], weights = nothing,
        states = nothing, params = nothing, rng::AbstractRNG = Random.default_rng(),
        t0 = 0.0, alg = Tsit5(), solve_kwargs...)
    p = collect(Float64, p)
    w = weights === nothing ? init_weights(rng, network) : collect(Float64, weights)
    np, nw = length(p), length(w)
    states = states === nothing ? [Symbol(:u, i) for i in eachindex(u0)] :
             collect(Symbol, states)
    params = params === nothing ? [Symbol(:p, i) for i in 1:np] : collect(Symbol, params)
    length(params) == np ||
        throw(ArgumentError("`params` has $(length(params)) names but p has $np entries"))
    wnames = [Symbol(:nn, i) for i in 1:nw]
    clash = intersect(params, wnames)
    isempty(clash) ||
        throw(ArgumentError("parameter names $clash are reserved for the network weights"))
    rhs = if f === nothing
        (u, P, t) -> network(u, view(P, (np + 1):(np + nw)))
    else
        (u, P, t) -> f(u, view(P, 1:np), t, x -> network(x, view(P, (np + 1):(np + nw))))
    end
    ode = ODEModel(rhs, u0, vcat(p, w); states, params = vcat(params, wnames), t0, alg,
        solve_kwargs...)
    return NeuralODEModel(network, ode, np)
end

Base.show(io::IO, m::NeuralODEModel) = print(io, "NeuralODEModel(", nstates(m),
    " states ", statenames(m), ", ", m.nparams, " parameters + ", nweights(m),
    " network weights)")

"Number of network weights of a [`NeuralODEModel`](@ref)."
nweights(m::NeuralODEModel) = length(m.ode.prob.p) - m.nparams

"""
    weightnames(model::NeuralODEModel)

Names of the network weights (`:nn1, :nn2, …`), to pass to [`calibrate!`](@ref) or
to `Twin(...; estimate)`.
"""
weightnames(m::NeuralODEModel) = [Symbol(:nn, i) for i in 1:nweights(m)]

"""
    networkweights(model::NeuralODEModel, p = nominal parameters)

The network weights in a parameter vector, for example
`networkweights(model, twin.p)` after calibration.
"""
networkweights(m::NeuralODEModel, p = m.ode.prob.p) = p[(m.nparams + 1):end]

# Everything else is the ODE model's behaviour.
for f in (:nstates, :statenames, :user_parameters, :nominal_state, :nominal_params,
          :initial_time)
    @eval $f(m::NeuralODEModel) = $f(m.ode)
end
state_index(m::NeuralODEModel, s) = state_index(m.ode, s)
getparam(m::NeuralODEModel, p, s) = getparam(m.ode, p, s)
setparam!(m::NeuralODEModel, p, s, v) = setparam!(m.ode, p, s, v)
propagate(m::NeuralODEModel, args...; kwargs...) = propagate(m.ode, args...; kwargs...)
advance(m::NeuralODEModel, args...; kwargs...) = advance(m.ode, args...; kwargs...)
trajectory(m::NeuralODEModel, args...; kwargs...) = trajectory(m.ode, args...; kwargs...)
