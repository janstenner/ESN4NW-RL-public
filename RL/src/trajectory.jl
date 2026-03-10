Base.haskey(t::AbstractTrajectory, s::Symbol) = s in keys(t)
Base.isempty(t::AbstractTrajectory) = all(k -> isempty(t[k]), keys(t))

function Base.empty!(t::AbstractTrajectory)
    for k in keys(t)
        empty!(t[k])
    end
end

function Base.push!(t::AbstractTrajectory; kwargs...)
    for (k, v) in kwargs
        push!(t[k], v)
    end
end

function Base.pop!(t::AbstractTrajectory)
    for k in keys(t)
        pop!(t[k])
    end
end

function Base.show(io::IO, t::AbstractTrajectory)
    println(io, "Trajectory of $(length(keys(t))) traces:")
    for k in keys(t)
        show(io, k)
        println(io, " $(summary(t[k]))")
    end
end

#####
# Common Keys
#####

const SART = (:state, :action, :reward, :terminated)
const SARTS = (:state, :action, :reward, :terminated, :next_state)
const SARTSA = (:state, :action, :reward, :terminated, :next_state, :next_action)
const SLART = (:state, :legal_actions_mask, :action, :reward, :terminated)
const SLARTSL = (
    :state,
    :legal_actions_mask,
    :action,
    :reward,
    :terminated,
    :next_state,
    :next_legal_actions_mask,
)
const SLARTSLA = (
    :state,
    :legal_actions_mask,
    :action,
    :reward,
    :terminated,
    :next_state,
    :next_legal_actions_mask,
    :next_action,
)

const SARTTS = (
    :state,
    :action,
    :reward,
    :terminated,
    :truncated,
    :next_state,
)





#####
# Trajectory
#####

"""
    Trajectory(;[trace_name=trace_container]...)

A simple wrapper of `NamedTuple`.
Define our own type here to avoid type piracy with `NamedTuple`
"""
struct Trajectory{T} <: AbstractTrajectory
    traces::T
end

Trajectory(; kwargs...) = Trajectory(values(kwargs))

@forward Trajectory.traces Base.getindex, Base.keys

Base.merge(a::Trajectory, b::Trajectory) = Trajectory(merge(a.traces, b.traces))
Base.merge(a::Trajectory, b::NamedTuple) = Trajectory(merge(a.traces, b))
Base.merge(a::NamedTuple, b::Trajectory) = Trajectory(merge(a, b.traces))




function CircularArrayTrajectory(; capacity, kwargs...)
    Trajectory(map(values(kwargs)) do x
        CircularArrayBuffer{eltype(first(x))}(last(x)..., capacity)
    end)
end


function CircularVectorTrajectory(; capacity, kwargs...)
    Trajectory(map(values(kwargs)) do x
        CircularVectorBuffer{x}(capacity)
    end)
end




function Base.length(
    t::Trajectory
)
    x = t[:terminated]
    size(x, ndims(x))
end






abstract type AbstractSampler{traces} end

"""
    sample([rng=Random.GLOBAL_RNG], trajectory, sampler, [traces=Val(keys(trajectory))])

!!! note
    Here we return a copy instead of a view:
    1. Each sample is independent of the original `trajectory` so that `trajectory` can be updated async.
    2. [Copy is not always so bad](https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad).
"""
function sample(t::AbstractTrajectory, sampler::AbstractSampler)
    sample(Random.GLOBAL_RNG, t, sampler)
end

#####
## BatchSampler
#####

mutable struct BatchSampler{traces} <: AbstractSampler{traces}
    batch_size::Int
    cache::Any
    rng::Any
end

BatchSampler(batch_size::Int; cache = nothing, rng = Random.GLOBAL_RNG) =
    BatchSampler{SARTSA}(batch_size, cache, rng)
BatchSampler{T}(batch_size::Int; cache = nothing, rng = Random.GLOBAL_RNG) where {T} =
    BatchSampler{T}(batch_size, cache, rng)

(s::BatchSampler)(t::AbstractTrajectory) = sample(s.rng, t, s)

function sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t)-1, s.batch_size)
    fetch!(s, t, inds)
    inds, s.cache
end

function fetch!(s::BatchSampler, t::AbstractTrajectory, inds::Vector{Int})

    batch = NamedTuple{keys(t)}(consecutive_view(t[x], inds) for x in keys(t))

    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end


# custom sample function

function pde_sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler, number_actuators::Int)
    inds = rand(rng, 1:length(t)-number_actuators, s.batch_size)
    pde_fetch!(s, t, inds, number_actuators)
    inds, s.cache
end

function pde_fetch!(s::BatchSampler, t::Trajectory, inds::Vector{Int}, number_actuators::Int)

    batch = NamedTuple{SARTS}((
        (consecutive_view(t[x], inds) for x in SART)...,
        consecutive_view(t[:state], inds .+ number_actuators),
    ))

    
    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end


function pde_sample(rng::AbstractRNG, t::AbstractTrajectory, s::BatchSampler)
    inds = rand(rng, 1:length(t), s.batch_size)
    pde_fetch!(s, t, inds)
    inds, s.cache
end

function pde_fetch!(s::BatchSampler, t::Trajectory, inds::Vector{Int})

    batch = NamedTuple{SARTS}((
        (consecutive_view(t[x], inds) for x in SART)...,
        consecutive_view(t[:state], inds .+ 1),
    ))

    
    if isnothing(s.cache)
        s.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(s.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
end