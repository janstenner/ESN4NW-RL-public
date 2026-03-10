

Base.@kwdef struct CustomNeuralNetworkApproximator{M,O}
    model::M
    optimizer::O = nothing
end

# some model may accept multiple inputs
(app::CustomNeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

@forward CustomNeuralNetworkApproximator.model Flux.testmode!,
Flux.trainmode!,
Flux.params,
device


update!(app::CustomNeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)

copyto!(dest::CustomNeuralNetworkApproximator, src::CustomNeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, Flux.params(src))







#####
# ActorCritic
#####

"""
    ActorCritic(;actor, critic, optimizer=ADAM())

The `actor` part must return logits (*Do not use softmax in the last layer!*), and the `critic` part must return a state value.
"""
Base.@kwdef mutable struct ActorCritic{A,C}
    actor::A
    critic::C
    optimizer_actor = ADAM()
    optimizer_critic = ADAM()
    actor_state_tree = nothing
    critic_state_tree = nothing
end

# functor(x::ActorCritic) = (actor=x.actor, critic=x.critic), y -> ActorCritic(y.actor, y.critic, x.optimizer)

#update!(app::ActorCritic, gs) = Flux.Optimise.update!(app.optimizer, params(app), gs)

function Base.copyto!(dest::ActorCritic, src::ActorCritic)
    Flux.loadparams!(dest.actor, params(src.actor))
    Flux.loadparams!(dest.critic, params(src.critic))
end

@forward ActorCritic.critic device




#####
# GaussianNetwork (for PPO)
##### 

"""
     normlogpdf(μ, σ, x; ϵ = 1.0f-8)
GPU automatic differentiable version for the logpdf function of normal distributions.
Adding an epsilon value to guarantee numeric stability if sigma is exactly zero
(e.g. if relu is used in output layer).
"""
const log2π = log(2f0π)

function normlogpdf(μ, σ, x; ϵ = 1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end


"""
    GaussianNetwork(;pre=identity, μ, logσ, min_σ=0f0, max_σ=Inf32, normalizer = tanh)

Returns `μ` and `logσ` when called.  Create a distribution to sample from using
`Normal.(μ, exp.(logσ))`. `min_σ` and `max_σ` are used to clip the output from
`logσ`. Actions are normalized according to the specified normalizer function.
"""
Base.@kwdef struct GaussianNetwork{P,U,S,F}
    pre::P = identity
    μ::U
    logσ::S
    logσ_is_network::Bool = false
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
    normalizer::F = tanh
end

GaussianNetwork(pre, μ, logσ, min_σ=0.0f0, max_σ=Inf32, normalizer=tanh) = GaussianNetwork(pre, μ, logσ, min_σ, max_σ, normalizer)

Flux.@layer GaussianNetwork


# ---- constants + helpers (optional; du nutzt bereits normlogpdf) ----
const LOG2F = log(2f0)

# numerisch stabil: log|det d(tanh(u))/du| = sum 2*(log 2 - u - softplus(-2u))
log_jac_tanh(u) = 2f0 .* (LOG2F .- u .- softplus.(-2f0 .* u))

# ---- GaussianNetwork call mit Antithetic-Sampling ----
function (model::GaussianNetwork)(
    rng::AbstractRNG, s;
    is_sampling::Bool=false,
    is_return_log_prob::Bool=false,
    is_return_params::Bool=false,
    is_antithetic::Bool=false
)
    x = model.pre(s)

    if model.logσ_is_network
        μ, raw_logσ = model.μ(x), model.logσ(x)
    else
        μ, raw_logσ = model.μ(x), model.logσ
        if ndims(μ) >= 2
            raw_logσ = send_to_device(device(model.logσ), ones(Float32, 1, size(μ)[2])) .* model.logσ
        else
            raw_logσ = raw_logσ[:]
        end
    end

    # Zuerst per min_σ/max_σ, dann harter Safety-Clamp in Log-Space
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    logσ = clamp.(logσ, -20f0, 2f0)  # harte Ober-/Untergrenze für Varianz

    if !is_sampling
        return μ, deepcopy(logσ)
    end

    σ = exp.(logσ)

    noise = Zygote.ignore() do
        randn(rng, Float32, size(μ))
    end

    if is_antithetic
        # antithetische Paare im u-Space
        u_plus  = μ .+ σ .* noise
        u_minus = μ .- σ .* noise   # == 2μ - u_plus

        z_plus  = model.normalizer.(u_plus)
        z_minus = model.normalizer.(u_minus)

        if is_return_log_prob
            # logπ(u) = logN(u; μ,σ) - log|det J_tanh(u)| ; Summation über Action-Dim (dims=1)
            logp_plus  = sum(normlogpdf(μ, σ, u_plus)  .- log_jac_tanh(u_plus),  dims=1)
            logp_minus = sum(normlogpdf(μ, σ, u_minus) .- log_jac_tanh(u_minus), dims=1)

            if is_return_params
                return z_plus, logp_plus, z_minus, logp_minus, μ, deepcopy(logσ)
            else
                return z_plus, logp_plus, z_minus, logp_minus
            end
        else
            return z_plus, z_minus
        end

    else
        # gewöhnliches einzelnes Sample
        u = μ .+ σ .* noise
        z = model.normalizer.(u)

        if is_return_log_prob
            logp_π = sum(normlogpdf(μ, σ, u) .- log_jac_tanh(u), dims=1)
            if is_return_params
                return z, logp_π, μ, deepcopy(logσ)
            else
                return z, logp_π
            end
        else
            return z
        end
    end
end

# convenience-wrapper ohne rng (beibehaltener Default)
function (model::GaussianNetwork)(
    state; is_sampling::Bool=false, is_return_log_prob::Bool=false, is_antithetic::Bool=false
)
    model(Random.GLOBAL_RNG, state;
          is_sampling=is_sampling,
          is_return_log_prob=is_return_log_prob,
          is_antithetic=is_antithetic)
end