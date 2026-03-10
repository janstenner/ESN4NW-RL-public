function create_chain(;
    ns,
    na,
    use_gpu,
    is_actor,
    init,
    nna_scale,
    network_depth = 2,
    drop_middle_layer = nothing,
    fun = relu,
    tanh_end = false
)
    if !isnothing(drop_middle_layer)
        network_depth = drop_middle_layer ? 1 : 2
    end
    network_depth = max(1, Int(network_depth))

    nna_size_actor = Int(floor(10 * nna_scale))
    nna_size_critic = Int(floor(20 * nna_scale))

    if is_actor
        layers = Any[Dense(ns, nna_size_actor, fun; init = init)]
        for _ in 2:network_depth
            push!(layers, Dense(nna_size_actor, nna_size_actor, fun; init = init))
        end
        output_fun = tanh_end ? tanh : identity
        push!(layers, Dense(nna_size_actor, na, output_fun; init = init))
        n = Chain(layers...)
    else
        layers = Any[Dense(ns, nna_size_critic, fun; init = init)]
        for _ in 2:network_depth
            push!(layers, Dense(nna_size_critic, nna_size_critic, fun; init = init))
        end
        push!(layers, Dense(nna_size_critic, 1; init = init))
        n = Chain(layers...)
    end

    model = use_gpu ? n |> gpu : n

    model
end

function create_logσ(;logσ_is_network, ns, na, use_gpu, init, nna_scale, network_depth = 2, drop_middle_layer = nothing, fun = relu, start_logσ = 0.0)

    res = nothing

    if logσ_is_network
        res = create_chain(
            ns = ns,
            na = na,
            use_gpu = use_gpu,
            is_actor = true,
            init = init,
            nna_scale = nna_scale,
            network_depth = network_depth,
            drop_middle_layer = drop_middle_layer,
            fun = fun,
            tanh_end = false
        )
    else
        res = Matrix(Matrix(Float32.(ones(na) .* start_logσ)')')
    end

    res = use_gpu ? res |> gpu : res

    return res
end

function create_agent_ppo(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, learning_rate_critic = nothing, fun = relu, fun_critic = nothing, tanh_end = false, n_envs = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = true, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, noise = nothing, noise_scale = 90, clip_range_vf = nothing, dist = Normal, verbose = false)

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

    default_actor = if dist == Normal
        GaussianNetwork(
            μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, tanh_end = tanh_end),
            logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, start_logσ = start_logσ),
            logσ_is_network = logσ_is_network,
            max_σ = max_σ
        )
    elseif dist == Categorical
        # For discrete policies, the actor must output logits (no tanh).
        create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, tanh_end = false)
    else
        error("Unsupported distribution type $(dist). Use Normal or Categorical.")
    end

    action_schema =
        dist == Categorical ?
        (Int => (n_envs,)) :
        (Float32 => (size(action_space)[1], n_envs))

    Agent(
        policy = PPOPolicy(
            approximator = isnothing(approximator) ? ActorCritic(
                actor = default_actor,
                critic = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic),
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas)),
            ) : approximator,
            γ = y,
            λ = p,
            clip_range = clip_range,
            clip_range_vf = clip_range_vf,
            max_grad_norm = 0.5f0,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            dist = dist,
            rng = rng,
            update_freq = update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            noise = noise,
            noise_sampler = noise_sampler,
            noise_scale = noise_scale,
            verbose = verbose,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_envs),
                action = action_schema,
                action_log_prob = Float32 => (n_envs),
                reward = Float32 => (n_envs),
                terminated = Bool => (n_envs,),
                truncated = Bool => (n_envs,),
                next_state = Float32 => (size(state_space)[1], n_envs),
        ),
    )
end





mutable struct PPOPolicy{A<:ActorCritic,D,R} <: AbstractPolicy
    approximator::A
    γ::Float32
    λ::Float32
    clip_range::Float32
    clip_range_vf::Union{Nothing,Float32}
    max_grad_norm::Float32
    n_microbatches::Int
    n_epochs::Int
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    rng::R
    n_random_start::Int
    update_freq::Int
    update_step::Int
    clip1::Bool
    normalize_advantage::Bool
    start_steps
    start_policy
    target_kl
    noise
    noise_sampler
    noise_scale
    noise_step
    verbose::Bool
    last_action_log_prob::Vector{Float32}
    last_sigma::Vector{Float32}
    last_mu::Vector{Float32}
end

function PPOPolicy(;
    approximator,
    update_freq,
    n_random_start=0,
    update_step=0,
    γ=0.99f0,
    λ=0.95f0,
    clip_range=0.2f0,
    clip_range_vf = 0.2f0,
    max_grad_norm=0.5f0,
    n_microbatches=4,
    n_epochs=4,
    actor_loss_weight=1.0f0,
    critic_loss_weight=0.5f0,
    entropy_loss_weight=0.01f0,
    dist=Normal,
    rng=Random.GLOBAL_RNG,
    clip1 = false,
    normalize_advantage = normalize_advantage,
    start_steps = -1,
    start_policy = nothing,
    target_kl = 100.0,
    noise = nothing,
    noise_sampler = nothing,
    noise_scale = 90.0,
    noise_step = 0,
    verbose = false,
)
    PPOPolicy{typeof(approximator),dist,typeof(rng)}(
        approximator,
        γ,
        λ,
        clip_range,
        clip_range_vf,
        max_grad_norm,
        n_microbatches,
        n_epochs,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        rng,
        n_random_start,
        update_freq,
        update_step,
        clip1,
        normalize_advantage,
        start_steps,
        start_policy,
        target_kl,
        noise,
        noise_sampler,
        noise_scale,
        noise_step,
        verbose,
        [0.0],
        [0.0],
        [0.0],
    )
end

function prob(
    p::PPOPolicy{<:ActorCritic{<:GaussianNetwork},Normal},
    state::AbstractArray,
    mask,
)
    if p.update_step < p.n_random_start
        @error "todo"
    else
        μ, logσ =
            p.approximator.actor(send_to_device(device(p.approximator), state)) |>
            send_to_host
        StructArray{Normal}((μ, exp.(logσ)))
    end
end

function prob(p::PPOPolicy{<:ActorCritic,Categorical}, state::AbstractArray, mask)
    logits = p.approximator.actor(send_to_device(device(p.approximator), state))
    if ndims(logits) == 1
        logits = reshape(logits, :, 1)
    end
    if !isnothing(mask)
        mask_local = ndims(mask) == 1 ? reshape(mask, :, 1) : mask
        logits .+= ifelse.(mask_local, 0.0f0, typemin(Float32))
    end
    logits = logits |> softmax |> send_to_host
    if p.update_step < p.n_random_start
        [
            Categorical(fill(1 / length(x), length(x)); check_args=false) for
            x in eachcol(logits)
        ]
    else
        [Categorical(x; check_args=false) for x in eachcol(logits)]
    end
end

function prob(p::PPOPolicy, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::PPOPolicy, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::PPOPolicy)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1 && !(eltype(result) <: Integer)
        clamp!(result, -1.0, 1.0)
    end
    result
end


# !!! https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/533/files#r728920324
function (p::PPOPolicy)(env::AbstractEnv)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        if p isa PPOPolicy{<:ActorCritic,Categorical}
            action_dist = prob(p, env)
            if length(action_dist) == 1
                action = rand(p.rng, action_dist[1])
                log_p = [Float32(logpdf(action_dist[1], action))]
            else
                action = rand.(p.rng, action_dist)
                log_p = Float32.(vec(logpdf.(action_dist, action)))
            end
            p.last_action_log_prob = log_p
            p.last_mu = [0.0f0]
            p.last_sigma = [0.0f0]
            action
        else
            dist = prob(p, env)

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
end

function (agent::Agent{<:PPOPolicy})(env::MultiThreadEnv)

    if agent.policy.update_step <= agent.policy.start_steps
        agent.policy.start_policy(env)
    else
        dist = prob(agent.policy, env)
        action = rand.(agent.policy.rng, dist)
        if agent.policy.clip1 && !(eltype(action) <: Integer)
            clamp!(action, -1.0, 1.0)
        end
        logpdf_values = logpdf.(dist, action)
        action_log_prob =
            ndims(logpdf_values) == 2 ?
            vec(sum(logpdf_values, dims=1)) :
            vec(logpdf_values)
        EnrichedAction(action; action_log_prob=action_log_prob)
    end
end






function update!(
    trajectory::AbstractTrajectory,
    ::PPOPolicy,
    env::MultiThreadEnv,
    ::PreActStage,
    action::EnrichedAction,
)
    push!(
        trajectory;
        state=state(env),
        action=action.action,
        action_log_prob=action.action_log_prob
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    action_store =
        policy isa PPOPolicy{<:ActorCritic,Categorical} && action isa Integer ?
        Int[action] :
        action
    push!(
        trajectory;
        state=state(env),
        action=action_store,
        action_log_prob=policy.last_action_log_prob
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::PPOPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)[:]

    push!(trajectory[:reward], r)
    push!(trajectory[:terminated], is_terminated(env))
    push!(trajectory[:truncated], is_truncated(env))
    push!(trajectory[:next_state], state(env))
end

function update!(
    p::PPOPolicy,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    length(t) == 0 && return  # in the first update, only state & action are inserted into trajectory
    p.update_step += 1
    if p.update_step % p.update_freq == 0
        _update!(p, t)
    end
end

function update_IL(p::PPOPolicy, trajectory::AbstractTrajectory)
    n_total = length(trajectory)
    n_total == 0 && return

    window = min(p.update_freq, n_total)
    start_idx = rand(p.rng, 1:(n_total - window + 1))
    stop_idx = start_idx + window - 1

    action_full = trajectory[:action]
    action_window =
        ndims(action_full) == 3 ?
        action_full[:, :, start_idx:stop_idx] :
        ndims(action_full) == 2 ?
        action_full[:, start_idx:stop_idx] :
        action_full[start_idx:stop_idx]

    temp_trajectory = Trajectory(
        state = trajectory[:state][:, :, start_idx:stop_idx],
        action = action_window,
        action_log_prob = trajectory[:action_log_prob][:, start_idx:stop_idx],
        reward = trajectory[:reward][:, start_idx:stop_idx],
        terminated = trajectory[:terminated][:, start_idx:stop_idx],
        truncated = trajectory[:truncated][:, start_idx:stop_idx],
        next_state = trajectory[:next_state][:, :, start_idx:stop_idx],
    )

    _update!(p, temp_trajectory)
end





function _update!(p::PPOPolicy, t::Any)
    if p.verbose
        println("TRAIN!!!!!!!!")
    end
    
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
    
    D = device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_rollout = size(t[:terminated])

    microbatch_size = Int(floor(n_envs * n_rollout ÷ n_microbatches))

    n = length(t)
    states = to_device(t[:state])
    next_states = to_device(t[:next_state])


    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))

    values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)
    next_values = reshape(send_to_host(AC.critic(flatten_batch(next_states))), n_envs, :)

    advantages, returns = generalized_advantage_estimation(
        t[:reward],
        values,
        next_values,
        γ,
        λ;
        dims=2,
        terminated=t[:terminated],
        truncated=t[:truncated]
    )
    returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    advantages = to_device(advantages)

    # if p.normalize_advantage
    #     advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
    # end

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)

    stop_update = false

    actor_losses = Float32[]
    critic_losses = Float32[]

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a_host = collect(select_last_dim(actions_flatten, inds))
            a = to_device(a_host)
            a_discrete = eltype(a_host) <: Integer ? Int.(vec(a_host)) : nothing

            r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds]
            adv = vec(advantages)[inds]
            old_v = vec(values)[inds]

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end
            

            if isnothing(AC.actor_state_tree)
                AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor)
            end

            if isnothing(AC.critic_state_tree)
                AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
            end

            g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic
                v′ = critic(s) |> vec
                if actor isa GaussianNetwork
                    μ, logσ = actor(s)
                    
                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ), a), dims=1))
                    else
                        log_p′ₐ = normlogpdf(μ, exp.(logσ), a)
                    end
                    entropy_loss =
                        mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                else
                    # actor is assumed to return discrete logits
                    logit′ = actor(s)
                    if ndims(logit′) == 1
                        logit′ = reshape(logit′, :, 1)
                    end

                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)
                    if isnothing(a_discrete)
                        error("Categorical PPO expects integer actions in trajectory.")
                    end
                    n_actions = size(log_p′, 1)
                    one_hot_actions = to_device(Float32.(Flux.onehotbatch(a_discrete, 1:n_actions)))
                    log_p′ₐ = vec(sum(log_p′ .* one_hot_actions; dims=1))
                    entropy_loss = -mean(sum(p′ .* log_p′; dims=1))
                end
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

                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))

                if isnothing(clip_range_vf) || clip_range_vf == 0.0
                    critic_loss = mean((r .- v′) .^ 2)
                else
                    # clipped value function loss, from OpenAI SpinningUp implementation
                    Δ = v′ .- old_v
                    values_pred = old_v .+ clamp.(Δ, -clip_range_vf, clip_range_vf)
                    critic_loss = mean((r .- values_pred) .^ 2)
                end


                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss


                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                end

                loss
            end
            
            if !stop_update
                Flux.update!(AC.actor_state_tree, AC.actor, g_actor)
                Flux.update!(AC.critic_state_tree, AC.critic, g_critic)


            else
                break
            end

        end

        if stop_update
            break
        end
    end


    # mean_actor_loss = mean(abs.(actor_losses))
    # mean_critic_loss = mean(abs.(critic_losses))
    # println("---")
    # println(mean_actor_loss)
    # println(mean_critic_loss)
    

    # actor_factor = clamp(0.5/mean_actor_loss, 0.9, 1.1)
    # critic_factor = clamp(0.3/mean_critic_loss, 0.9, 1.1)
    # println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
    # println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
    # p.actor_loss_weight = w₁ * actor_factor
    # p.critic_loss_weight = w₂ * critic_factor

    # println("---")
end
