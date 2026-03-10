struct ZeroPolicy <: AbstractPolicy
    action_space
end

(p::ZeroPolicy)(env) = zeros(size(p.action_space))


#####
# Spaces
#####

export WorldSpace

"""
In some cases, we may not be interested in the action/state space.
One can return `WorldSpace()` to keep the interface consistent.
"""
struct WorldSpace{T} end

WorldSpace() = WorldSpace{Any}()

Base.in(x, ::WorldSpace{T}) where {T} = x isa T

export Space

"""
A wrapper to treat each element as a sub-space which supports:

- `Base.in`
- `Random.rand`
"""
struct Space{T}
    s::T
end

Base.:(==)(x::Space, y::Space) = x.s == y.s
Base.similar(s::Space, args...) = Space(similar(s.s, args...))
Base.getindex(s::Space, args...) = getindex(s.s, args...)
Base.setindex!(s::Space, args...) = setindex!(s.s, args...)
Base.size(s::Space) = size(s.s)
Base.length(s::Space) = length(s.s)
Base.iterate(s::Space, args...) = iterate(s.s, args...)

Random.rand(s::Space) = rand(Random.GLOBAL_RNG, s)

Random.rand(rng::AbstractRNG, s::Space) =
    map(s.s) do x
        rand(rng, x)
    end

Random.rand(rng::AbstractRNG, s::Space{<:Dict}) = Dict(k => rand(rng, v) for (k, v) in s.s)

function Base.in(X, S::Space)
    if length(X) == length(S.s)
        for (x, s) in zip(X, S.s)
            if x ∉ s
                return false
            end
        end
        return true
    else
        return false
    end
end

function Base.in(X::Dict, S::Space{<:Dict})
    if keys(X) == keys(S.s)
        for k in keys(X)
            if X[k] ∉ S.s[k]
                return false
            end
        end
        return true
    else
        return false
    end
end




"""
    flatten_batch(x::AbstractArray)

Merge the last two dimension.

# Example

```julia-repl
julia> x = reshape(1:12, 2, 2, 3)
2×2×3 reshape(::UnitRange{Int64}, 2, 2, 3) with eltype Int64:
[:, :, 1] =
 1  3
 2  4

[:, :, 2] =
 5  7
 6  8

[:, :, 3] =
  9  11
 10  12

julia> flatten_batch(x)
2×6 reshape(::UnitRange{Int64}, 2, 6) with eltype Int64:
 1  3  5  7   9  11
 2  4  6  8  10  12
```
"""
flatten_batch(x::AbstractArray) = reshape(x, size(x)[1:end-2]..., :)


"""
    consecutive_view(x::AbstractArray, inds; n_stack = nothing, n_horizon = nothing)

By default, it behaves the same with `select_last_dim(x, inds)`.
If `n_stack` is set to an int, then for each frame specified by `inds`,
the previous `n_stack` frames (including the current one) are concatenated as a new dimension.
If `n_horizon` is set to an int, then for each frame specified by `inds`,
the next `n_horizon` frames (including the current one) are concatenated as a new dimension.

# Example

```julia
julia> x = collect(1:5)
5-element Array{Int64,1}:
 1
 2
 3
 4
 5

julia> consecutive_view(x, [2,4])  # just the same with `select_last_dim(x, [2,4])`
2-element view(::Array{Int64,1}, [2, 4]) with eltype Int64:
 2
 4

julia> consecutive_view(x, [2,4];n_stack = 2)
2×2 view(::Array{Int64,1}, [1 3; 2 4]) with eltype Int64:
 1  3
 2  4

julia> consecutive_view(x, [2,4];n_horizon = 2)
2×2 view(::Array{Int64,1}, [2 4; 3 5]) with eltype Int64:
 2  4
 3  5

julia> consecutive_view(x, [2,4];n_horizon = 2, n_stack=2)  # note the order here, first we stack, then we apply the horizon
2×2×2 view(::Array{Int64,1}, [1 2; 2 3]

[3 4; 4 5]) with eltype Int64:
[:, :, 1] =
 1  2
 2  3

[:, :, 2] =
 3  4
 4  5
```

See also [Frame Skipping and Preprocessing for Deep Q networks](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)
to gain a better understanding of state stacking and n-step learning.
"""
select_last_dim(xs::AbstractArray{T,N}, inds) where {T,N} =
    @views xs[ntuple(_ -> (:), N - 1)..., inds]

select_last_frame(xs::AbstractArray{T,N}) where {T,N} = select_last_dim(xs, size(xs, N))

consecutive_view(
    cb::AbstractArray,
    inds::Vector{Int};
    n_stack = nothing,
    n_horizon = nothing,
) = consecutive_view(cb, inds, n_stack, n_horizon)

consecutive_view(cb::AbstractArray, inds::Vector{Int}, ::Nothing, ::Nothing) =
    select_last_dim(cb, inds)

consecutive_view(cb::AbstractArray, inds::Vector{Int}, n_stack::Int, ::Nothing) =
    select_last_dim(
        cb,
        reshape([i for x in inds for i in x-n_stack+1:x], n_stack, length(inds)),
    )

consecutive_view(cb::AbstractArray, inds::Vector{Int}, ::Nothing, n_horizon::Int) =
    select_last_dim(
        cb,
        reshape([i for x in inds for i in x:x+n_horizon-1], n_horizon, length(inds)),
    )

consecutive_view(cb::AbstractArray, inds::Vector{Int}, n_stack::Int, n_horizon::Int) =
    select_last_dim(
        cb,
        reshape(
            [j for x in inds for i in x:x+n_horizon-1 for j in i-n_stack+1:i],
            n_stack,
            n_horizon,
            length(inds),
        ),
    )






const VectorOrMatrix = Union{AbstractMatrix,AbstractVector}

function generalized_advantage_estimation(
    rewards::VectorOrMatrix,
    values::VectorOrMatrix,
    next_values::VectorOrMatrix,
    γ::T,
    λ::T;
    kwargs...,
) where {T<:Number}
    advantages = similar(rewards, promote_type(eltype(rewards), T))
    boot_returns = similar(rewards, promote_type(eltype(rewards), T))
    generalized_advantage_estimation!(advantages, boot_returns, rewards, values, next_values, γ, λ; kwargs...)
    
    advantages, boot_returns
end

generalized_advantage_estimation!(
    advantages,
    boot_returns,
    rewards,
    values,
    next_values,
    γ,
    λ;
    terminated = nothing,
    truncated = nothing,
    dims = :,
) = _generalized_advantage_estimation!(advantages, boot_returns, rewards, values, next_values, γ, λ, terminated, truncated, dims)

function _generalized_advantage_estimation!(
    advantages::AbstractMatrix,
    boot_returns::AbstractMatrix,
    rewards::AbstractMatrix,
    values::AbstractMatrix,
    next_values::AbstractMatrix,
    γ,
    λ,
    terminated::Nothing,
    truncated::Nothing,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (adv, ret, r, v, nv) in zip(
        eachslice(advantages, dims = dims),
        eachslice(boot_returns, dims = dims),
        eachslice(rewards, dims = dims),
        eachslice(values, dims = dims),
        eachslice(next_values, dims = dims),
    )
        _generalized_advantage_estimation!(adv, ret, r, v, nv, γ, λ, nothing, nothing)
    end
end


function _generalized_advantage_estimation!(
    advantages::AbstractMatrix,
    boot_returns::AbstractMatrix,
    rewards::AbstractMatrix,
    values::AbstractMatrix,
    next_values::AbstractMatrix,
    γ,
    λ,
    terminated,
    truncated,
    dims::Int,
)
    dims = ndims(rewards) - dims + 1
    for (adv, ret, r, v, nv, ter, trun) in zip(
        eachslice(advantages, dims = dims),
        eachslice(boot_returns, dims = dims),
        eachslice(rewards, dims = dims),
        eachslice(values, dims = dims),
        eachslice(next_values, dims = dims),
        eachslice(terminated, dims = dims),
        eachslice(truncated, dims = dims),
    )
        _generalized_advantage_estimation!(adv, ret, r, v, nv, γ, λ, ter, trun)
    end
end

_generalized_advantage_estimation!(
    advantages::AbstractVector,
    boot_returns::AbstractVector,
    rewards::AbstractVector,
    values::AbstractVector,
    next_values::AbstractVector,
    γ,
    λ,
    terminated,
    truncated,
    dims,
) = _generalized_advantage_estimation!(advantages, boot_returns, rewards, values, next_values, γ, λ, terminated, truncated)


# assuming rewards and advantages are Vector
function _generalized_advantage_estimation!(advantages, boot_returns, rewards, values, next_values, γ, λ, terminated, truncated)
    gae = 0.0f0
    T = length(rewards)

    for i in T:-1:1
        term = isnothing(terminated) ? false : terminated[i]
        trunc = isnothing(truncated)  ? false : truncated[i]
        done = term || trunc                 # Reset-Grenze
        bootstrap = !term                    # Timeout/trunc => true, echte Termination => false

        delta = rewards[i] + γ * next_values[i] * bootstrap - values[i]
        gae   = delta + γ * λ * (!done) * gae

        advantages[i] = gae

        next_ret =
            (i == T)    ? (next_values[i] * bootstrap) :
            term        ? 0.0f0 :
            trunc       ? next_values[i] :
                          boot_returns[i+1]

        boot_returns[i] = rewards[i] + γ * next_ret
    end

    return advantages, boot_returns
end



global_norm(gs::Zygote.Grads, ps::Zygote.Params) =
    sqrt(sum(mapreduce(x -> x^2, +, gs[p]) for p in ps))

function clip_by_global_norm!(gs::Zygote.Grads, ps::Zygote.Params, clip_norm::Float32)
    gn = global_norm(gs, ps)
    if clip_norm <= gn
        for p in ps
            gs[p] .*= clip_norm / max(clip_norm, gn)
        end
    end
    gn
end