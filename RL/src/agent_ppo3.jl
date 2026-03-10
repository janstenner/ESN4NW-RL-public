
Base.@kwdef mutable struct PPOPolicy3{A,D,R} <: AbstractPolicy
    approximator::A
    γ::Float32 = 0.99f0
    λ::Float32 = 0.99f0
    clip_range::Float32 = 0.2f0
    clip_range_vf::Union{Nothing,Float32} = 0.2f0
    n_microbatches::Int = 1
    actorbatch_size = nothing
    n_epochs::Int = 5
    actor_loss_weight = 1.0f0
    critic_loss_weight = 0.5f0
    entropy_loss_weight = 0.01f0
    adaptive_weights::Bool = true
    rng::R = Random.GLOBAL_RNG
    update_freq::Int = 2048
    critic_frozen_update_freq::Int = 4
    actor_update_freq::Int = 1
    update_step::Int = 0
    clip1::Bool = false
    normalize_advantage::Bool = true
    start_steps = -1
    start_policy = nothing
    target_kl = 100.0
    noise = nothing
    noise_sampler = nothing
    noise_scale = 90.0
    noise_step = 0
    fear_factor = 0.1f0
    fear_scale = 0.4f0
    new_loss::Bool = true
    use_popart::Bool = false
    critic_frozen_factor::Float32 = 0.1f0
    λ_targets = 0.7f0
    n_targets = 100
    mm = ModulationModule()
    critic2_takes_action::Bool = true
    use_exploration_module::Bool = false
    use_whole_delta_targets::Bool = false
    C2bar = nothing
    critic3_trajectory = nothing

    antithetic_mean_samples = 4
    zero_mean_tether_factor = 0.8f0

    off_policy_update_freq = 0
    off_policy_batch_size = 256

    verbose = false

    last_action_log_prob::Vector{Float32} = [0.0f0]
    last_sigma::Vector{Float32} = [0.0f0]
    last_mu::Vector{Float32} = [0.0f0]
end




Base.@kwdef mutable struct ActorCritic3{A,C,C2}
    actor::A
    critic::C # value function approximator
    critic_frozen::C
    critic2::C2 # action value function approximator minus r
    critic2_frozen::C2
    critic3 = nothing
    optimizer_actor = ADAM()
    optimizer_sigma = ADAM()
    optimizer_critic = ADAM()
    optimizer_critic2 = ADAM()
    optimizer_critic3 = ADAM()
    actor_state_tree = nothing
    sigma_state_tree = nothing
    critic_state_tree = nothing
    critic2_state_tree = nothing
    critic3_state_tree = nothing
end

@forward ActorCritic3.critic device




# Pre-activation residual block: LN -> GELU -> Dense -> GELU -> Dense + skip
struct ResMLPBlock
    ln1::LayerNorm
    d1::Dense
    ln2::LayerNorm
    d2::Dense
end
ResMLPBlock(width::Int) = ResMLPBlock(
    LayerNorm(width), Dense(width, width, gelu),
    LayerNorm(width), Dense(width, width, gelu),
)
(m::ResMLPBlock)(x) = x .+ m.d2(m.ln2(m.d1(m.ln1(x))))



function create_agent_ppo3(;action_space, state_space, use_gpu, rng, y, p, update_freq = 2000, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, learning_rate_critic = nothing, fun = relu, fun_critic = nothing, tanh_end = false, n_envs = 1, clip1 = false, n_epochs = 10, n_microbatches = 1, actorbatch_size=nothing, normalize_advantage = false, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, adaptive_weights = false, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, clip_range_vf = 0.2f0, noise = nothing, noise_scale = 90, fear_factor = 0.1, fear_scale = 0.4, new_loss = true, dist = Normal, critic_frozen_update_freq = 4, actor_update_freq = 1, critic2_takes_action = true, use_popart = false, critic_frozen_factor = 0.1f0, λ_targets = 0.7f0, n_targets = 100, use_critic3 = false, use_exploration_module = false, use_whole_delta_targets = false, antithetic_mean_samples = 4, zero_mean_tether_factor = 0.8f0, verbose = false, trajectory_size = 100_000, off_policy_update_freq = 0, off_policy_batch_size = 256, )

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    !isnothing(drop_middle_layer)        &&  (network_depth = drop_middle_layer ? 1 : 2)
    !isnothing(drop_middle_layer_critic) &&  (network_depth_critic = drop_middle_layer_critic ? 1 : 2)
    isnothing(network_depth_critic)      &&  (network_depth_critic = network_depth)
    network_depth = max(1, Int(network_depth))
    network_depth_critic = max(1, Int(network_depth_critic))
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(learning_rate_critic)     &&  (learning_rate_critic = learning_rate)

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    if noise == "perlin"
        noise_sampler = perlin_2d(; seed=Int(floor(rand(rng) * 1e12)))
    else
        noise_sampler = nothing
    end

    if use_whole_delta_targets
        critic2_takes_action = true
    end

    mm = ModulationModule()

    if off_policy_update_freq == 0
        trajectory_size = update_freq
    end

    critic = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic, popart = use_popart)

    csize = 128
    # critic = Chain(
    #     Dense(ns, csize, gelu),              # widening stem
    #     (ResMLPBlock(csize) for _ in 1:4)...,
    #     LayerNorm(csize),                        # final pre-act norm stabilizes the head
    #     Dense(csize, 1)                          # linear, unbounded value head
    # )
    critic_frozen = deepcopy(critic)

    critic2 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic, is_critic2 = critic2_takes_action, popart = use_popart)

    # critic2 = Chain(
    #     Dense(ns+na, csize, gelu),              # widening stem
    #     (ResMLPBlock(csize) for _ in 1:4)...,
    #     LayerNorm(csize),                        # final pre-act norm stabilizes the head
    #     Dense(csize, 1)                          # linear, unbounded value head
    # )
    critic2_frozen = deepcopy(critic2)

    if use_critic3
        #critic3 is now baseline
        critic3 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic, is_critic2 = false, popart = use_popart)

        critic3_trajectory = CircularArrayTrajectory(;
                capacity = update_freq * 5,
                state = Float32 => (size(state_space)[1], n_envs),
        )

        C2bar = deepcopy(critic2)
    else
        critic3 = nothing
        critic3_trajectory = nothing
        C2bar = nothing
    end

    approximator = isnothing(approximator) ? ActorCritic3(
                actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                ),
                critic = critic,
                critic_frozen = critic_frozen,
                critic2 = critic2,
                critic2_frozen = critic2_frozen,
                critic3 = critic3,
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
                optimizer_sigma = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
                optimizer_critic2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
                optimizer_critic3 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
            ) : approximator

    Agent(
        policy = PPOPolicy3{typeof(approximator),dist,typeof(rng)}(
            approximator = approximator,
            γ = y,
            λ = p,
            clip_range = clip_range,
            clip_range_vf = clip_range_vf,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actorbatch_size = actorbatch_size,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            adaptive_weights = adaptive_weights,
            rng = rng,
            update_freq = update_freq,
            critic_frozen_update_freq = critic_frozen_update_freq,
            actor_update_freq = actor_update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            noise = noise,
            noise_sampler = noise_sampler,
            noise_scale = noise_scale,
            fear_factor = fear_factor,
            fear_scale = fear_scale,
            new_loss = new_loss,
            use_popart = use_popart,
            critic_frozen_factor = critic_frozen_factor,
            λ_targets = λ_targets,
            n_targets = n_targets,
            mm = mm,
            critic2_takes_action = critic2_takes_action,
            use_exploration_module = use_exploration_module,
            use_whole_delta_targets = use_whole_delta_targets,
            C2bar = C2bar,
            critic3_trajectory = critic3_trajectory,

            verbose = verbose,

            antithetic_mean_samples = antithetic_mean_samples,
            zero_mean_tether_factor = zero_mean_tether_factor,

            off_policy_update_freq = off_policy_update_freq,
            off_policy_batch_size = off_policy_batch_size,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = trajectory_size,
                state = Float32 => (size(state_space)[1], n_envs),
                action = Float32 => (size(action_space)[1], n_envs),
                action_log_prob = Float32 => (n_envs),
                reward = Float32 => (n_envs),
                explore_mod = Float32 => (n_envs),
                terminated = Bool => (n_envs,),
                truncated = Bool => (n_envs,),
                next_state = Float32 => (size(state_space)[1], n_envs),
        ),
    )
end








function prob(
    p::PPOPolicy3{<:ActorCritic3{<:GaussianNetwork},Normal},
    state::AbstractArray,
    mask,
)
    μ, logσ = p.approximator.actor(send_to_device(device(p.approximator), state)) |> send_to_host

    StructArray{Normal}((μ, exp.(logσ)))
end

function prob(p::PPOPolicy3{<:ActorCritic3,Categorical}, state::AbstractArray, mask)
    logits = p.approximator.actor(send_to_device(device(p.approximator), state))
    if !isnothing(mask)
        logits .+= ifelse.(mask, 0.0f0, typemin(Float32))
    end
    logits = logits |> softmax |> send_to_host
    
    [Categorical(x; check_args=false) for x in eachcol(logits)]
end

function prob(p::PPOPolicy3, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::PPOPolicy3, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::PPOPolicy3)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end



function (p::PPOPolicy3)(env::AbstractEnv)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        dist = prob(p, env)

        modulation_value = p.mm(p.use_exploration_module)


        dist.σ .*= modulation_value
        #dist.σ .+= modulation_value * 0.25

        # if p.clip1
        #     clamp!(dist.μ, -1.0, 1.0)
        # end

        if isnothing(p.noise)
            action = rand.(p.rng, dist)
        else
            norm_factor = float(pi / 20) #* 5
            p.noise_step += 1
            noise = [CoherentNoise.sample(p.noise_sampler, p.noise_step/p.noise_scale, float(π+2*i))/norm_factor for i in 1:size(dist.μ, 2)]
            action = dist.μ + dist.σ .* noise'
        end

        if p.clip1
            clamp!(action, -1.0, 1.0)
        end

        # put the last action log prob behind the clip
        
        ###p.last_action_log_prob = vec(sum(logpdf.(dist, action), dims=1))

        if ndims(action) == 2
            log_p = vec(sum(normlogpdf(dist.μ, dist.σ, action), dims=1))
        else
            log_p = normlogpdf(dist.μ, dist.σ, action)
        end

        p.last_action_log_prob = log_p
        p.last_mu = dist.μ[:]
        p.last_sigma = dist.σ[:]

        action
    end
end








function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy3,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(
        trajectory;
        state = state(env),
        action = action,
        action_log_prob = policy.last_action_log_prob,
        explore_mod = policy.mm.last,
    )

    if !isnothing(policy.critic3_trajectory)
        push!(
        policy.critic3_trajectory;
        state = state(env),
    )
    end
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy3,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)[:]

    push!(trajectory[:reward], r)
    push!(trajectory[:terminated], is_terminated(env))
    push!(trajectory[:truncated], is_truncated(env))
    push!(trajectory[:next_state], state(env))
    #push!(trajectory[:next_values], policy.approximator.critic(send_to_device(device(policy.approximator), env.state)) |> send_to_host)
end

function update!(
    p::PPOPolicy3,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    length(t) == 0 && return  # in the first update, only state & action are inserted into trajectory
    p.update_step += 1


    number_actuators = size(t[:action])[2]

    if p.off_policy_update_freq != 0 && p.update_step % p.off_policy_update_freq == 0
        inds, batch = pde_sample(p.rng, t, BatchSampler{SARTS}(p.off_policy_batch_size), number_actuators)
        _update_off_policy!(p, batch)
    end

    if p.update_step % p.update_freq == 0
        _update!(p, t)
    end
end

function update_IL(p::PPOPolicy3, trajectory::AbstractTrajectory)
    n_total = length(trajectory)
    n_total == 0 && return

    window = min(p.update_freq, n_total)
    start_idx = rand(p.rng, 1:(n_total - window + 1))
    stop_idx = start_idx + window - 1

    temp_trajectory = Trajectory(
        state = trajectory[:state][:, :, start_idx:stop_idx],
        action = trajectory[:action][:, :, start_idx:stop_idx],
        action_log_prob = trajectory[:action_log_prob][:, start_idx:stop_idx],
        reward = trajectory[:reward][:, start_idx:stop_idx],
        explore_mod = trajectory[:explore_mod][:, start_idx:stop_idx],
        terminated = trajectory[:terminated][:, start_idx:stop_idx],
        truncated = trajectory[:truncated][:, start_idx:stop_idx],
        next_state = trajectory[:next_state][:, :, start_idx:stop_idx],
    )

    _update!(p, temp_trajectory)
end










function nstep_targets(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32 = 0.99f0;
    n::Int = 3
) :: Vector{Float32}
    T = length(rewards)
    @assert length(dones) == T && length(next_values) == T
    targets = similar(rewards)

    @inbounds for t in 1:T
        g::Float32 = 0f0
        discount::Float32 = 1f0
        hit_done = false
        last_idx = t # wird im Loop auf den letzten wirklich verwendeten Index gesetzt

        # bis zu n Rewards einsammeln (oder bis Episodenende/Sequenzende)
        for k in 0:(n-1)
            idx = t + k
            if idx > T
                break
            end
            g += discount * rewards[idx]
            last_idx = idx
            if dones[idx]                # Episode endet hier → kein Bootstrapping
                hit_done = true
                break
            end
            discount *= γ                # γ^(k+1)
        end

        # Bootstrapping NUR wenn kein done getroffen wurde.
        # Wichtig: mit dem tatsächlich letzten Index bootstrappen (last_idx),
        # nicht blind mit t+n-1.
        if !hit_done && last_idx ≥ t && last_idx ≤ T && !dones[last_idx]
            # discount ist hier bereits γ^m mit m = Anzahl gesammelter Rewards
            # next_values[last_idx] entspricht V(s_{last_idx+1})
            g += discount * next_values[last_idx]
        end

        targets[t] = g
    end
    return targets
end

function nstep_targets(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    n::Int=3
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = nstep_targets(rewards[i, :], dones[i, :], next_values[i, :], γ; n=n)
    end

    return results
end


function td_lambda_targets(
    rewards::Vector{Float32},
    terminated::Vector{Bool},
    truncated::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    λ::Float32=0.7f0
) :: Vector{Float32}
    T = length(rewards)
    targets = similar(rewards)
    Gλ = 0f0

    for t in T:-1:1
        term = terminated[t]
        trunc = truncated[t]
        done = term || trunc
        bootstrap = !term  # trunc => true, echte termination => false

        G_next = done ? (next_values[t] * bootstrap) : Gλ

        Gλ = rewards[t] + γ * ((1f0 - λ) * next_values[t] + λ * G_next) * bootstrap
        targets[t] = Gλ
    end
    return targets
end

function td_lambda_targets(
    rewards::AbstractMatrix,
    terminated::AbstractMatrix,
    truncated::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = td_lambda_targets(rewards[i, :], terminated[i, :], truncated[i, :], next_values[i, :], γ; λ=λ)
    end

    return results
end


function lambda_truncated_targets(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
    n::Int=3
) :: Vector{Float32}

    # berechne alle k‑Step‑Returns mit next_values
    Gs = [nstep_targets(rewards, dones, next_values, γ; n=k) for k in 1:n]

    T = length(rewards)
    targets = similar(rewards)
    for t in 1:T
        sum_part = 0f0
        for k in 1:(n-1)
            sum_part += (1f0-λ)*λ^(k-1) * Gs[k][t]
        end
        sum_part += λ^(n-1) * Gs[n][t]
        targets[t] = sum_part
    end
    return targets
end

function lambda_truncated_targets(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
    n::Int=3
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = lambda_truncated_targets(rewards[i, :], dones[i, :], next_values[i, :], γ; λ=λ, n=n)
    end

    return results
end



# -- PPO3 special target functions -- #

function nstep_targets_ppo3(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32 = 0.99f0;
    n::Int = 3
) :: Vector{Float32}
    T = length(rewards)
    @assert length(dones) == T && length(next_values) == T
    targets = similar(rewards)

    @inbounds for t in 1:T
        g::Float32 = 0f0
        discount::Float32 = γ
        hit_done = false
        last_idx = t  # wird im Loop auf den letzten wirklich verwendeten Index gesetzt

        # bis zu n Rewards einsammeln (oder bis Episodenende/Sequenzende)
        for k in 1:(n-1)
            idx = t + k
            if idx > T
                break
            end
            g += discount * rewards[idx]
            last_idx = idx
            if dones[idx]                # Episode endet hier → kein Bootstrapping
                hit_done = true
                break
            end
            discount *= γ                # γ^(k+1)
        end

        # Bootstrapping NUR wenn kein done getroffen wurde.
        # Wichtig: mit dem tatsächlich letzten Index bootstrappen (last_idx),
        # nicht blind mit t+n-1.
        if !hit_done && last_idx ≥ t && last_idx ≤ T && !dones[last_idx]
            # discount ist hier bereits γ^m mit m = Anzahl gesammelter Rewards
            # next_values[last_idx] entspricht V(s_{last_idx+1})
            g += discount * next_values[last_idx]
        end

        targets[t] = g
    end
    return targets
end

function nstep_targets_ppo3(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    n::Int=3
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = nstep_targets_ppo3(rewards[i, :], dones[i, :], next_values[i, :], γ; n=n)
    end

    return results
end

function lambda_truncated_targets_ppo3(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
    n::Int=3
) :: Vector{Float32}

    # berechne alle k‑Step‑Returns mit next_values
    Gs = [nstep_targets_ppo3(rewards, dones, next_values, γ; n=k) for k in 1:n]

    T = length(rewards)
    targets = similar(rewards)
    for t in 1:T
        sum_part = 0f0
        for k in 1:(n-1)
            sum_part += (1f0-λ)*λ^(k-1) * Gs[k][t]
        end
        sum_part += λ^(n-1) * Gs[n][t]
        targets[t] = sum_part
    end
    return targets
end

function lambda_truncated_targets_ppo3(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
    n::Int=3
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = lambda_truncated_targets_ppo3(rewards[i, :], dones[i, :], next_values[i, :], γ; λ=λ, n=n)
    end

    return results
end

function special_targets_ppo3(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    values::Vector{Float32},
    next_values::Vector{Float32},
    γ::Float32 = 0.99f0
) :: Vector{Float32}
    T = length(rewards)
    @assert length(dones) == T && length(values) == T && length(next_values) == T
    targets = similar(rewards)

    @inbounds for t in 1:T
        g::Float32 = 0f0
        hit_done = false

        g += rewards[t]
        
        if dones[t]
            hit_done = true
        else
            #g += γ * rewards[t + 1]
            g += γ * next_values[t]
        end
        
        g -= values[t]
            

        targets[t] = g
    end
    return targets
end

function special_targets_ppo3(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    values::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = special_targets_ppo3(rewards[i, :], dones[i, :], values[i, :], next_values[i, :], γ)
    end

    return results
end

function td_lambda_targets_ppo3(
    rewards::Vector{Float32},
    dones::Vector{Bool},
    values::Vector{Float32},
    next_values::Vector{Float32},
    γ::Float32=0.99f0;
    λ::Float32=0.7f0
) :: Vector{Float32}
    T = length(rewards)
    targets = similar(rewards)
    Gλ = 0f0
    for t in T:-1:1
        # für Schritt t ist next_values[t] = V(s_{t+1})
        #Gλ = rewards[t] + γ * ((1f0-λ)*next_values[t] + λ*Gλ) * (1 - dones[t]) - values[t]

        Gλ = ((1f0-λ)*next_values[t] + λ*Gλ) * (1 - dones[t])
        targets[t] = Gλ

        Gλ *= γ
        Gλ += rewards[t]
    end
    return targets
end

function td_lambda_targets_ppo3(
    rewards::AbstractMatrix,
    dones::AbstractMatrix,
    values::AbstractMatrix,
    next_values::AbstractMatrix,
    γ::Float32=0.99f0;
    λ::Float32=0.7f0,
) :: AbstractMatrix

    results = zeros(Float32, size(rewards))

    for i in 1:size(rewards, 1)
        results[i, :] = td_lambda_targets_ppo3(rewards[i, :], dones[i, :], values[i, :], next_values[i, :], γ; λ=λ)
    end

    return results
end


function antithetic_mean(actor, critic, states; K = 16)
    μ, ℓσ   = actor(states)

    Σ = exp.(ℓσ)

    acc  = zeros(Float32, 1, size(μ)[2]) 

    for k in 1:K
        ϵ   = randn(Float32, size(μ))

        a_plus  = μ .+ Σ .* ϵ
        a_minus = μ .- Σ .* ϵ

        y_plus  = send_to_host(critic(vcat(states, a_plus)))
        y_minus = send_to_host(critic(vcat(states, a_minus)))

        acc .+= (y_plus .+ y_minus)
    end
    mean_critic = acc ./ (2*K)                            # ≈ E_a C2(s,a)

    mean_critic
end





function check_state_trees(p::PPOPolicy3)
    # just to make sure the state trees are initialized

    AC = p.approximator

    reset_optimizers = false #(p.update_step / p.update_freq) % 400 == 0
    start_optimizers = isnothing(AC.actor_state_tree) || isnothing(AC.sigma_state_tree) || isnothing(AC.critic_state_tree) || isnothing(AC.critic2_state_tree)

    if start_optimizers || reset_optimizers
        if p.verbose
            println("________________________________________________________________________")
            println("Reset Optimizers")
            println("________________________________________________________________________")
        end
        AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor.μ)
        AC.sigma_state_tree = Flux.setup(AC.optimizer_sigma, AC.actor.logσ)
        AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
        AC.critic2_state_tree = Flux.setup(AC.optimizer_critic2, AC.critic2)

        Optimisers.adjust!(AC.critic_state_tree.layers[end]; lambda = 0.0)
        Optimisers.adjust!(AC.critic2_state_tree.layers[end]; lambda = 0.0)

        if !isnothing(AC.critic3)
            AC.critic3_state_tree = Flux.setup(AC.optimizer_critic3, AC.critic3)
            Optimisers.adjust!(AC.critic3_state_tree.layers[end]; lambda = 0.0)
        end
    end

    return
end


function _update!(p::PPOPolicy3, t::Any; update_actor = true, update_critic = true, time_step_interval = nothing, microbatch_size = nothing)


    rng = p.rng
    AC = p.approximator
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    clip_range_vf = p.clip_range_vf
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight
    D = RL.device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_trajectory = size(t[:terminated])
    n_rollout = min(p.update_freq, n_trajectory)


    if isnothing(time_step_interval)
        global valid_indices = collect(n_trajectory-n_rollout+1:n_trajectory)
    else
        # Get indices where time steps fall within the interval
        time_steps = t[:state][end, 1, :]
        global valid_indices = findall(x -> time_step_interval[1] <= x <= time_step_interval[2], time_steps)
    end

    n_samples = length(valid_indices)


    if isnothing(microbatch_size)
        microbatch_size = Int(floor(n_samples ÷ n_microbatches))
    else
        microbatch_size = min(microbatch_size, n_samples)
        n_microbatches = Int(floor(n_samples ÷ microbatch_size))
    end

    actorbatch_size = p.actorbatch_size


    states = to_device(t[:state][:,:,valid_indices])
    actions = to_device(t[:action][:,:,valid_indices])

    states_flatten_on_host = flatten_batch(select_last_dim(t[:state][:,:,valid_indices], 1:n_samples))

    values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)


    #mus = AC.actor.μ(states_flatten_on_host)
    #offsets = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), mus) )) , n_envs, :)

    # advantages = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), flatten_batch(actions)) )) , n_envs, :) - offsets

    critic2_input = p.critic2_takes_action ? vcat(flatten_batch(states), flatten_batch(actions)) : flatten_batch(states)

    critic2_values = reshape(send_to_host( AC.critic2( critic2_input ) ) , n_envs, :)

    mus = AC.actor.μ(states_flatten_on_host)
    offset_input = vcat(flatten_batch(states), mus)
    offsets = reshape(send_to_host( AC.critic2( offset_input )) , n_envs, :)

    rewards = collect(to_device(t[:reward][:,valid_indices]))
    terminated = collect(to_device(t[:terminated][:,valid_indices]))
    truncated = collect(to_device(t[:truncated][:,valid_indices]))


    

    if p.use_whole_delta_targets
        #gae_deltas = rewards .+ critic2_values .* (1 .- terminated) .- values

        if !isnothing(AC.critic3)

            offset_input = flatten_batch(states)
            offsets = reshape(send_to_host( AC.critic3( offset_input )) , n_envs, :)

            gae_deltas = critic2_values - offsets
        else
            mean_c2 = antithetic_mean(AC.actor, AC.critic2, states_flatten_on_host; K = p.antithetic_mean_samples)

            gae_deltas = critic2_values - mean_c2
        end

        advantages, returns = generalized_advantage_estimation(
            gae_deltas,
            zeros(Float32, size(gae_deltas)),
            zeros(Float32, size(gae_deltas)),
            γ,
            λ;
            dims=2,
            terminated=terminated,
            truncated=truncated
        )
    else
        advantages, returns = generalized_advantage_estimation(
            rewards,
            values,
            critic2_values,
            γ,
            λ;
            dims=2,
            terminated=terminated,
            truncated=truncated
        )
    end

    # returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    advantages = to_device(advantages)

    # if p.normalize_advantage
    #     advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
    # end

    positive_advantage_indices = findall(>(0), vec(advantages))


    actions_flatten = flatten_batch(select_last_dim(t[:action][:,:,valid_indices], 1:n_samples))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob][:,valid_indices]), 1:n_samples)
    explore_mod = to_device(t[:explore_mod][:,valid_indices])

    stop_update = false

    actor_losses = Float32[]
    critic_losses = Float32[]
    critic2_losses = Float32[]
    entropy_losses = Float32[]



    
    check_state_trees(p)


    # AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
    # AC.critic2_state_tree = Flux.setup(AC.optimizer_critic2, AC.critic2)
    # Optimisers.adjust!(AC.critic_state_tree.layers[end]; lambda = 0.0)
    # Optimisers.adjust!(AC.critic2_state_tree.layers[end]; lambda = 0.0)

    next_states = to_device(flatten_batch(t[:next_state][:,:,valid_indices]))

    v_ref = AC.critic_frozen( flatten_batch(states) )[:] 

    q_ref = AC.critic2_frozen( critic2_input )[:] 


    next_values = reshape( AC.critic( next_states ), n_envs, :)
    
    #targets = lambda_truncated_targets(rewards, terminated, next_values, γ; λ = p.λ_targets, n = p.n_targets)[:]
    targets = td_lambda_targets(rewards, terminated, truncated, next_values, γ; λ = p.λ_targets)[:]

    #targets for critic2 now below


    collector = BatchQuantileCollector()
    

    

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, collect(1:n_samples))
        #rand_inds_actor = shuffle!(rng, Vector(1:n_envs*n_rollout-actorbatch_size))

        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            #inds_actor = collect(rand_inds_actor[i]:rand_inds_actor[i]+actorbatch_size-1)
            if isnothing(actorbatch_size)
                inds_actor = inds
            else
                inds_actor = inds[1:clamp(actorbatch_size, 1, length(inds))]
            end


            # DRIFT LOG
            # Auf Actor-Batch (inds_actor) – nach Vorwärtsrechnungen
            # A_cent = gae_deltas[inds_actor]

            # mean_A = mean(A_cent)
            # std_A  = std(A_cent) + 1f-8

            # # Policy-gemittelte Abweichung (billig via μ)
            # μ      = AC.actor.μ(states_flatten_on_host)
            # A_mu   = reshape(send_to_host(AC.critic2(vcat(flatten_batch(states), μ))), n_envs, :) .- offsets
            # E_A    = mean(vec(A_mu)[inds_actor])

            # @info "adv_centered" mean_A=mean_A std_A=std_A E_pi_A=E_A

            #inds = positive_advantage_indices

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            s_actor = to_device(collect(select_last_dim(states_flatten_on_host, inds_actor)))
            a = to_device(collect(select_last_dim(actions_flatten, inds_actor)))
            exp_m = to_device(collect(select_last_dim(explore_mod, inds_actor)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            #r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds_actor]
            adv = vec(advantages)[inds_actor]

            tar = vec(targets)[inds]

            old_v = vec(values)[inds]

            

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            # s_neg = sample_negatives_far(s)

            g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic
                v′ = critic(s) |> vec

                # nv′ = AC.critic(ns) |> vec
                # nv = critic2(vcat(s,a)) |> vec

                μ, logσ = actor(s_actor)

                σ = (exp.(logσ) .* exp_m) #.+ (exp_m .* 0.25)

                if ndims(a) == 2
                    log_p′ₐ = vec(sum(normlogpdf(μ, σ, a), dims=1))
                else
                    log_p′ₐ = normlogpdf(μ, σ, a)
                end

                #clamp!(log_p′ₐ, log(1e-8), Inf)

                entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                
                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl
                        if p.verbose
                            println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        end
                        stop_update = true
                    end
                end

                
                fear = (ratio .- 1).^2 .* p.fear_factor


                if p.new_loss
                    actor_loss_values = ((ratio .* adv) - fear)  #.* exp_m[:]
                    actor_loss = -mean(actor_loss_values)
                else
                    surr1 = ratio .* adv
                    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))
                end


                if isnothing(clip_range_vf) || clip_range_vf == 0.0
                    values_pred = v′
                else
                    # clipped value function loss, from OpenAI SpinningUp implementation
                    Δ = v′ .- old_v
                    values_pred = old_v .+ clamp.(Δ, -clip_range_vf, clip_range_vf)
                end



                
                bellman = mean(((tar .- values_pred) .^ 2))
                fr_term = mean((values_pred .- v_ref[inds]) .^ 2)
                critic_loss = bellman + p.critic_frozen_factor * fr_term # .* exp_m[:])


                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss 


                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)

                    update!(collector, ratio, adv; p=0.9, within_batch_weighted=true)
                end

                loss
            end
            
            if !stop_update
                if (p.update_step / p.update_freq) % p.actor_update_freq == 0
                    if update_actor
                        Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
                        Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
                    end
                end
                if update_critic
                    Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
                end
            else
                break
            end

            if p.use_popart && update_critic
                update!(AC.critic.layers[end], tar)
            end

        end

        if stop_update
            break
        end
    end

    #critic2_input = p.critic2_takes_action ? vcat(next_states, AC.actor.μ(next_states)) : next_states
    #next_critic2_values = reshape( AC.critic2( critic2_input ), n_envs, :)
    #targets_critic2 = lambda_truncated_targets_ppo3(rewards, terminated, next_values, γ)[:]
    values = reshape( AC.critic( states ), n_envs, :)
    next_values = reshape( AC.critic( next_states ), n_envs, :)
    #targets_critic2 = special_targets_ppo3(rewards, terminated, values, next_values, γ)[:]
    #@show size(rewards), size(terminated), size(values), size(next_values)



    if p.use_whole_delta_targets
        #@show size(rewards), size(terminated), size(values), size(next_values)
        targets_critic2 = rewards + next_values .* γ .* (1 .- terminated) - values
    else
        #targets_critic2 = td_lambda_targets_ppo3(rewards, terminated, values, next_values, γ; λ = p.λ_targets)[:]
        targets_critic2 = next_values .* (1 .- terminated)
    end


    # critic 2 zero mean tether

    c2 = AC.critic2 #p.C2bar

    # K    = 16 #p.baseline_mc_samples   # z.B. 4 oder 8
    # μ, ℓσ   = AC.actor(states_flatten_on_host)

    # Σ = exp.(ℓσ)

    # acc  = zeros(Float32, size(targets_critic2))  # [act_dim, batch]
    # for k in 1:K
    #     ϵ   = randn(Float32, size(μ))

    #     a_plus  = μ .+ Σ .* ϵ
    #     a_minus = μ .- Σ .* ϵ

    #     y_plus  = send_to_host(c2(vcat(states_flatten_on_host, a_plus)))
    #     y_minus = send_to_host(c2(vcat(states_flatten_on_host, a_minus)))

    #     acc .+= (y_plus .+ y_minus)
    # end
    # mean_c2 = acc ./ (2*K)

    mean_c2 = antithetic_mean(AC.actor, c2, states_flatten_on_host; K = p.antithetic_mean_samples)
        

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, collect(1:n_samples))

        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))

            # nv′ = vec(next_values)[inds]
            # rew = vec(rewards)[inds]
            # ter = vec(terminated)[inds]

            #tar = rew + γ * nv′ .* (1 .- ter)
            tar = vec(targets_critic2)[inds]

            old_v2 = vec(critic2_values)[inds]

            tether = mean((vec(mean_c2)[inds]).^2)

            critic2_input = p.critic2_takes_action ? vcat(s, a) : s

            g_critic2 = Flux.gradient(AC.critic2) do critic2
                v2′ = critic2(critic2_input) |> vec

                if isnothing(clip_range_vf) || clip_range_vf == 0.0
                    values_pred2 = v2′
                else
                    # clipped value function loss, from OpenAI SpinningUp implementation
                    Δ = v2′ .- old_v2
                    values_pred2 = old_v2 .+ clamp.(Δ, -clip_range_vf, clip_range_vf)
                end


                bellman = mean(((tar .- values_pred2) .^ 2))
                fr_term = mean((values_pred2 .- q_ref[inds]) .^ 2)
                critic2_loss = bellman + p.critic_frozen_factor * fr_term + p.zero_mean_tether_factor * tether # .* exp_m[:]


                ignore() do
                    push!(critic2_losses, w₂ * critic2_loss)
                end

                loss = w₂ * critic2_loss

                loss
            end

            if update_critic
                Flux.update!(AC.critic2_state_tree, AC.critic2, g_critic2[1])

                if p.use_popart
                    update!(AC.critic2.layers[end], tar) 
                end
            end

        end
    end


    if !isnothing(AC.critic3)

        # don't use target c2 for now

        # τ = 0.99
        # for (pb, p) in zip(Flux.params(p.C2bar), Flux.params(AC.critic2))
        #     pb .= τ .* pb .+ (1f0 - τ) .* p
        # end

        states_c3 = flatten_batch(select_last_dim(p.critic3_trajectory[:state], 1:n_samples))

        c2 = AC.critic2 #p.C2bar

        # K    = 16 #p.baseline_mc_samples   # z.B. 4 oder 8
        # μ, ℓσ   = AC.actor(states_c3)

        # Σ = exp.(ℓσ)

        # acc  = zeros(Float32, 1, size(μ)[2]) 

        # for k in 1:K
        #     ϵ   = randn(Float32, size(μ))

        #     a_plus  = μ .+ Σ .* ϵ
        #     a_minus = μ .- Σ .* ϵ

        #     y_plus  = send_to_host(c2(vcat(states_c3, a_plus)))
        #     y_minus = send_to_host(c2(vcat(states_c3, a_minus)))

        #     acc .+= (y_plus .+ y_minus)
        # end
        # mean_c2 = acc ./ (2*K)                            # ≈ E_a C2(s,a)

        mean_c2 = antithetic_mean(AC.actor, c2, states_c3; K = p.antithetic_mean_samples)

        targets_critic3 = reshape(mean_c2, n_envs, :) .* (1 .- terminated)

        #targets_critic3 = rewards + γ * critic2_values .* (1 .- terminated) - values

        for epoch in 1:n_epochs

            rand_inds = shuffle!(rng, Vector(1:size(states_c3)[2]))

            for i in 1:n_microbatches

                inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

                s = to_device(collect(select_last_dim(states_c3, inds)))

                tar = vec(targets_critic3)[inds]

                critic3_input = s

                g_critic3 = Flux.gradient(AC.critic3) do critic3
                    v3′ = critic3(critic3_input) |> vec

                    values_pred3 = v3′

                    bellman = mean(((tar .- values_pred3) .^ 2))
                    critic3_loss = bellman

                    loss = w₂ * critic3_loss

                    loss
                end

                if update_critic
                    Flux.update!(AC.critic3_state_tree, AC.critic3, g_critic3[1])

                    if p.use_popart
                        update!(AC.critic3.layers[end], tar)
                    end
                end

            end
        end
    end


    #println(p.update_step / p.update_freq)

    if (p.update_step / p.update_freq) % p.critic_frozen_update_freq == 0
        if update_critic
            if p.verbose
                println("CRITIC FROZEN UPDATE")
            end
            AC.critic_frozen = deepcopy(AC.critic)
            AC.critic2_frozen = deepcopy(AC.critic2)
        end
    end


    # everything here is just magnitude (abs), not real mean

    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic_loss = mean(abs.(critic_losses))
    mean_critic2_loss = mean(abs.(critic2_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    # mean_logσ_regularization_loss = mean(abs.(logσ_regularization_losses))
    # mean_critic_regularization_loss = mean(abs.(critic_regularization_losses))
    q = finalize(collector; p_over_epochs=0.9, weighted=true)
    
    if p.verbose
        println("---")
        println("mean actor loss: $(mean_actor_loss)")
        println("mean critic loss: $(mean_critic_loss)")
        println("mean critic2 loss: $(mean_critic2_loss)")
        println("mean entropy loss: $(mean_entropy_loss)")
        # println("mean logσ regularization loss: $(mean_logσ_regularization_loss)")
        # println("mean critic regularization loss: $(mean_critic_regularization_loss)")

        # q.q_eps   : 0.9-Quantil der (pro Batch) 0.9-Quantile von |r-1|
        # q.q_adv   : 0.9-Quantil der (pro Batch) 0.9-Quantile von |A|
        # q.wq_eps  : A-gewichtetes 0.9-Quantil über die Batch-Quantile von |r-1|

        println("0.9-Quantil excitement: $(q.q_adv)")
        println("weighted 0.9-Quantil |r-1|: $(q.wq_eps)")
    end



    if p.adaptive_weights && p.new_loss && (p.update_step / p.update_freq) % 4 == 0

        old_fear_factor = deepcopy(p.fear_factor)

        A_ref = q.q_adv                          # robustes |A|-Quantil
        eps_meas = q.wq_eps
        eps_target = 0.1                         # Zielwert für |r-1|

        # Guardrails / Fallbacks
        if !isfinite(eps_meas) || eps_meas <= 0
            eps_meas = eps_target                # Startfall oder Degenerat
        end
        if !isfinite(A_ref) || A_ref <= 0
            A_ref = 1.0                          # konservativer Fallback
        end

        # Baseline λ* aus Größenordnungs-Match nahe r≈1
        λ_star = 0.35 * (A_ref / eps_target)

        λ_prev = p.fear_factor
        gamma = 1.0
        beta = 0.9

        lambda_min = 1e-3
        lambda_max = 1e2

        # Regler-Update
        factor = (eps_meas / eps_target)^gamma
        λ_raw = (1 - beta) * λ_prev + beta * λ_star * factor

        # Clamping & Sanity
        λ_next = clamp(λ_raw, lambda_min, lambda_max)

        # polyak update fear_factor
        p.fear_factor = λ_next


        # println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
        # println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
        # println("changing entropy weight from $(w₃) to $(w₃*entropy_factor)")
        # println("changing logσ regularization weight from $(w₅) to $(w₅*logσ_regularization_factor)")
        # println("changing critic regularization weight from $(w₄) to $(w₄*critic_regularization_factor)")
        if p.verbose
            println("changing fear factor from $(old_fear_factor) to $(λ_next)")
        end

        # println("current critic_target is $(p.critic_target)")

        # p.actor_loss_weight = w₁ * actor_factor
        # p.critic_loss_weight = w₂ * critic_factor
        # p.entropy_loss_weight = w₃ * entropy_factor
        # p.logσ_regularization_loss_weight = w₅ * logσ_regularization_factor
        # p.critic_regularization_loss_weight = w₄ * critic_regularization_factor

        
    end

    if p.verbose
        println("---")
    end
end


function _update_off_policy!(p::PPOPolicy3, batch;)

    check_state_trees(p)

    AC = p.approximator
    γ = p.γ
    w₂ = p.critic_loss_weight

    D = RL.device(AC)
    s, a, r, t, next_states = send_to_device(D, batch)

    n_envs = size(t)[1]

    values = reshape(send_to_host(AC.critic(flatten_batch(s))), n_envs, :)
    next_values = reshape(send_to_host(AC.critic(flatten_batch(next_states))), n_envs, :)

    targets_critic = r + next_values .* γ .* (1 .- t)
    targets_critic2 = r + next_values .* γ .* (1 .- t) - values

    critic2_input = p.critic2_takes_action ? vcat(flatten_batch(s), flatten_batch(a)) : flatten_batch(s)
    v_ref = AC.critic_frozen( flatten_batch(s) )[:] 
    q_ref = AC.critic2_frozen( critic2_input )[:] 

    g_critic = Flux.gradient(AC.critic) do critic
        v′ = critic(s) |> vec

        values_pred = v′

        
        bellman = mean(((targets_critic[:] .- values_pred) .^ 2))
        fr_term = mean((values_pred .- v_ref) .^ 2)
        critic_loss = bellman + p.critic_frozen_factor * fr_term # .* exp_m[:])


        loss = w₂ * critic_loss

        loss
    end
    
    Flux.update!(AC.critic_state_tree, AC.critic, g_critic[1])

    if p.use_popart
        update!(AC.critic.layers[end], targets_critic[:])
    end


    g_critic2 = Flux.gradient(AC.critic2) do critic2
        v′ = critic2(critic2_input) |> vec

        values_pred = v′

        
        bellman = mean(((targets_critic2[:] .- values_pred) .^ 2))
        fr_term = mean((values_pred .- q_ref) .^ 2)
        critic2_loss = bellman + p.critic_frozen_factor * fr_term # .* exp_m[:])


        loss = w₂ * critic2_loss

        loss
    end
    
    Flux.update!(AC.critic2_state_tree, AC.critic2, g_critic2[1])

    if p.use_popart
        update!(AC.critic2.layers[end], targets_critic2[:])
    end
end
