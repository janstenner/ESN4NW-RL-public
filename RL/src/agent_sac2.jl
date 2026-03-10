
Base.@kwdef mutable struct SACPolicy2 <: AbstractPolicy
    actor::GaussianNetwork
    qnetwork1
    qnetwork2
    target_qnetwork1
    target_qnetwork2


    optimizer_actor = ADAM()
    optimizer_qnetwork1 = ADAM()
    optimizer_qnetwork2 = ADAM()
    actor_state_tree = nothing
    qnetwork1_state_tree = nothing
    qnetwork2_state_tree = nothing


    action_space::Space
    state_space::Space

    γ::Float32 =0.99f0
    τ::Float32 =0.005f0
    α::Float32 =0.2f0

    log_α =[log(0.2f0)]
    optimizer_log_α = ADAM()
    log_α_state_tree = nothing


    batch_size::Int =32
    start_steps::Int =-1
    start_policy = nothing
    update_after::Int =1000
    update_freq::Int =50
    update_loops::Int = 1
    automatic_entropy_tuning::Bool =true
    lr_alpha::Float32 =0.0003f0
    target_entropy::Float32 =-1.0f0
    update_step::Int =0
    rng =Random.GLOBAL_RNG
    device_rng =Random.GLOBAL_RNG
    use_popart = false

    fear_factor = 0.1f0
    on_policy_update_freq = 2500
    λ_targets = 0.7f0
    target_frac = 0.3f0
    verbose::Bool = false

    antithetic_mean_samples::Int = 16
    on_policy_n_batches::Int = 64
    on_policy_epochs::Int = 3

    # Logging
    last_reward_term::Float32 =0.0f0
    last_entropy_term::Float32 =0.0f0
    last_actor_loss::Float32 =0.0f0
    last_critic1_loss::Float32 =0.0f0
    last_critic2_loss::Float32 =0.0f0
    #last_log_alpha::Float32 =0.0f0
    last_q1_mean::Float32 =0.0f0
    last_q2_mean::Float32 =0.0f0
    last_target_q_mean::Float32 =0.0f0
    last_mean_minus_log_pi::Float32 =0.0f0
end


function create_agent_sac2(;action_space, state_space, use_gpu = false, rng, y, t =0.005f0, a =0.2f0, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, learning_rate_critic = nothing, fun = gelu, fun_critic = nothing, tanh_end = false, n_agents = 1, logσ_is_network = false, batch_size = 32, start_steps = -1, start_policy = nothing, update_after = 1000, update_freq = 50, update_loops = 1, max_σ = 7.0f0, min_σ = 2f-9, clip_grad = 0.5, start_logσ = 0.0, betas = (0.9, 0.999), trajectory_length = 10_000, automatic_entropy_tuning = true, lr_alpha = nothing, target_entropy = nothing, use_popart = false, on_policy_update_freq = 2500, λ_targets= 0.7f0, fear_factor = 0.1f0, target_frac = 0.3f0, verbose = false, antithetic_mean_samples = 16, on_policy_n_batches = 64, on_policy_epochs = 3)

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

    isnothing(target_entropy)           &&  (target_entropy = -Float32(na))

    isnothing(lr_alpha)                 && (lr_alpha = Float32(learning_rate))


    qnetwork1 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic, is_critic2 = true, popart = use_popart)
    target_qnetwork1 = deepcopy(qnetwork1)

    qnetwork2 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic, is_critic2 = true, popart = use_popart)
    target_qnetwork2 = deepcopy(qnetwork2)


    Agent(
        policy = SACPolicy2(
           actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ,
                    min_σ = min_σ,
                ),
            qnetwork1 = qnetwork1,
            qnetwork2 = qnetwork2,
            target_qnetwork1 = target_qnetwork1,
            target_qnetwork2 = target_qnetwork2,


            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
            optimizer_qnetwork1 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
            optimizer_qnetwork2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),

            action_space = action_space,
            state_space = state_space,
            
            γ = y,
            τ = t,
            α = a,
            log_α = [log(a)],
            optimizer_log_α = Optimisers.Adam(lr_alpha),

            batch_size = batch_size,
            start_steps = start_steps,
            start_policy = start_policy,
            update_after = update_after,
            update_freq = update_freq,
            update_loops = update_loops,
            automatic_entropy_tuning = automatic_entropy_tuning,
            lr_alpha = lr_alpha,
            target_entropy = target_entropy,
            rng = rng,
            device_rng = rng,
            use_popart = use_popart,

            fear_factor = fear_factor,
            on_policy_update_freq = on_policy_update_freq,
            λ_targets = λ_targets,
            target_frac = target_frac,
            verbose = verbose,

            antithetic_mean_samples = antithetic_mean_samples,
            on_policy_n_batches = on_policy_n_batches,
            on_policy_epochs = on_policy_epochs,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = trajectory_length,
                state = Float32 => (size(state_space)[1], n_agents),
                action = Float32 => (size(action_space)[1], n_agents),
                reward = Float32 => (n_agents),
                terminated = Bool => (n_agents,),
                truncated = Bool => (n_agents,),
                next_state = Float32 => (size(state_space)[1], n_agents),
        ),
    )
end



# TODO: handle Training/Testing mode
function (p::SACPolicy2)(env)

    if p.update_step <= p.start_steps
        action = p.start_policy(env)
        if(size(action[1]) != ())
            action = reduce(hcat, action)
        end
        action
    else
        # D = device(p.actor)
        # s = send_to_device(D, state(env))
        # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
        # trainmode:
        s = state(env)
        action = p.actor(p.device_rng, s; is_sampling=true)
        # action = dropdims(action, dims=ndims(action)) # Single action vec, drop second dim
        # send_to_host(action)

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "mean" action
        # action = dropdims(p.actor(s)[1], dims=2)

        action
    end
end








function update!(
    trajectory::AbstractTrajectory,
    policy::SACPolicy2,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    policy.update_step += 1
    
    push!(
        trajectory;
        state = state(env),
        action = action,
        #action_log_prob = policy.last_action_log_prob,
        #explore_mod = policy.mm.last,
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::SACPolicy2,
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
    p::SACPolicy2,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    # if length(size(p.action_space)) == 2
    #     number_actuators = size(p.action_space)[2]
    # else
    #     #mono case 
    #     number_actuators = 1
    # end

    length(t) > p.update_after || return

    
    
    if p.update_step % p.on_policy_update_freq == 0
        on_policy_update(p,t)
    end
    
    if p.update_step % p.update_freq == 0
        for i = 1:p.update_loops
            inds, batch = sample(p.rng, t, BatchSampler{SARTTS}(p.batch_size))
            update!(p, batch)
        end
    end
end





# ∥∇_a Q_min(s,a)∥ 
function grad_a_q_norms(qnet1, qnet2, s_batch, a_batch)
    qmin_sum(a_) = sum(min.(qnet1(vcat(s_batch, a_)), qnet2(vcat(s_batch, a_))))
    g = Zygote.gradient(qmin_sum, a_batch)[1]   # same shape as a_batch
    mags = sqrt.(sum(abs2, g; dims=1))[:]        # per-sample ‖∇_a Q_min‖
    mags
end


# ----- Q-basierte Maske für SAC: großes Q => kleine w -----
function q_rank_mask(q_batch::AbstractVector{<:Real};
                     target_frac::Float32=0.30f0, sharpness::Float32=20f0)
    q = Float32.(q_batch)
    B = length(q)
    r = (ordinalrank(q; rev=true) .- 1f0) ./ max(1f0, B-1f0)  # r=0 bei größtem Q
    @. 1f0 / (1f0 + exp(-sharpness*(r - target_frac))) |> Float32
end



# Anteil-Controller
function adjust_fear_factor(fear_factor, frac_changed; target=0.30f0, up=1.1f0, down=0.9f0, min = 1f-8, max = 1f8)
        
    new_factor = frac_changed > target ? fear_factor*up : fear_factor*down

    clamp(new_factor, min, max)
end


function check_state_trees(p::SACPolicy2)
    # just to make sure the state trees are initialized

    if isnothing(p.actor_state_tree) || isnothing(p.qnetwork1_state_tree) || isnothing(p.qnetwork2_state_tree)
        if p.verbose
            println("________________________________________________________________________")
            println("Reset Optimizers")
            println("________________________________________________________________________")
        end
        p.actor_state_tree = Flux.setup(p.optimizer_actor, p.actor)
        p.qnetwork1_state_tree = Flux.setup(p.optimizer_qnetwork1, p.qnetwork1)
        p.qnetwork2_state_tree = Flux.setup(p.optimizer_qnetwork2, p.qnetwork2)

        Optimisers.adjust!(p.qnetwork1_state_tree.layers[end]; lambda = 0.0)
        Optimisers.adjust!(p.qnetwork2_state_tree.layers[end]; lambda = 0.0)

        p.log_α_state_tree = Flux.setup(p.optimizer_log_α, p.log_α)
    end

    return
end


function antithetic_mean_sac2(p, states, α; use_grad_a_q_norms = false, K = 16)
    μ, logσ = p.actor(p.device_rng, states)

    acc_mu  = zeros(Float32, 1, size(μ,2), size(μ,3)) 
    acc_logp = zeros(Float32, 1, size(μ,2), size(μ,3))
    acc_mags  = zeros(Float32, size(μ,2)* size(μ,3)) 

    if isdefined(p, :antithetic_mean_samples)
        K = p.antithetic_mean_samples
    end

    for k in 1:K
        a_plus, logp_π_plus, a_minus, logp_π_minus = p.actor(p.device_rng, states; is_sampling=true, is_return_log_prob=true, is_antithetic = true)


        if use_grad_a_q_norms

            acc_mags .+= grad_a_q_norms(p.qnetwork1, p.qnetwork2, states, a_plus)
            acc_mags .+= grad_a_q_norms(p.qnetwork1, p.qnetwork2, states, a_minus)

        else

            acc_logp .+= logp_π_plus .+ logp_π_minus


            y_plus1  = send_to_host(p.target_qnetwork1(vcat(states, a_plus)))
            y_minus1 = send_to_host(p.target_qnetwork1(vcat(states, a_minus)))

            y_plus2  = send_to_host(p.target_qnetwork2(vcat(states, a_plus)))
            y_minus2 = send_to_host(p.target_qnetwork2(vcat(states, a_minus)))


            acc_mu .+= min.(y_plus1, y_plus2) .- α .* logp_π_plus
            acc_mu .+= min.(y_minus1, y_minus2) .- α .* logp_π_minus
        end
    end


    if use_grad_a_q_norms
        return acc_mags
    else
        acc_mu ./= 2*K
        acc_logp ./= 2*K

        return acc_mu, acc_logp
    end
end



function on_policy_update(p::SACPolicy2, traj::AbstractTrajectory; whole_trajectory = false)

    check_state_trees(p)

    if !whole_trajectory
        n_samples = p.on_policy_update_freq
        s = send_to_device(device(p.qnetwork1), traj[:state][:,:,end-n_samples:end-1])
        a = send_to_device(device(p.qnetwork1), traj[:action][:,:,end-n_samples:end-1])
        r = send_to_device(device(p.qnetwork1), traj[:reward][:,end-n_samples:end-1])
        ter = send_to_device(device(p.qnetwork1), traj[:terminated][:,end-n_samples:end-1])
        trun = send_to_device(device(p.qnetwork1), traj[:truncated][:,end-n_samples:end-1])
        next_states = send_to_device(device(p.qnetwork1), traj[:next_state][:,:,end-n_samples:end-1])
    else
        n_samples = length(traj)
        s = send_to_device(device(p.qnetwork1), traj[:state])
        a = send_to_device(device(p.qnetwork1), traj[:action])
        r = send_to_device(device(p.qnetwork1), traj[:reward])
        ter = send_to_device(device(p.qnetwork1), traj[:terminated])
        trun = send_to_device(device(p.qnetwork1), traj[:truncated])
        next_states = send_to_device(device(p.qnetwork1), traj[:next_state])
        #next_states[:,:, end] = zeros(Float32, size(s, 1), size(s, 2))  # terminal state
    end
    
    

    γ, τ, α = p.γ, p.τ, p.α


    acc_mu, acc_logp = antithetic_mean_sac2(p, next_states, α)
    
    logp_π′ = acc_logp

    next_values = acc_mu

    # if whole_trajectory
    #     next_values[:,:, end] .*= 0.0f0     # terminal states
    # end

    n_envs = size(ter, 1)
    next_values = reshape( next_values, n_envs, :)
    
    targets = td_lambda_targets(r, ter, trun, next_values, γ; λ = p.λ_targets)

    n_total = length(targets)
    n_batches = max(1, min(p.on_policy_n_batches, n_total))
    microbatch_size = Int(floor(n_total ÷ n_batches))

    D = device(p.qnetwork1)
    to_device(x) = send_to_device(D, x)

    states_flatten_on_host = flatten_batch(send_to_host(s))
    actions_flatten_on_host = flatten_batch(send_to_host(a))
    targets_flatten_on_host = vec(send_to_host(targets))

    # on policy actor update
    if p.target_frac < 1.0f0
        target_frac = p.target_frac
        τ_change = 3f-4
        μ_before, logσ_before = p.actor(p.device_rng, s)
        μ_before_flatten_on_host = flatten_batch(send_to_host(μ_before))
        logσ_before_flatten_on_host = flatten_batch(send_to_host(logσ_before))
    end

    for epoch in 1:p.on_policy_epochs
        rand_inds = shuffle!(p.rng, Vector(1:n_total))
        for i in 1:n_batches
            first_ind = (i - 1) * microbatch_size + 1
            first_ind > n_total && break
            last_ind = i == n_batches ? n_total : min(i * microbatch_size, n_total)
            inds = rand_inds[first_ind:last_ind]

            s_batch = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a_batch = to_device(collect(select_last_dim(actions_flatten_on_host, inds)))
            target_batch = to_device(Float32.(targets_flatten_on_host[inds]))
            target_batch_popart = reshape(target_batch, 1, :)

            q_input_batch = vcat(s_batch, a_batch)

            q_grad_1 = Flux.gradient(p.qnetwork1) do qnetwork1
                q1 = dropdims(qnetwork1(q_input_batch), dims=1)

                ignore_derivatives() do
                    p.last_q1_mean = mean(q1)
                    p.last_critic1_loss = Flux.mse(q1, target_batch)
                end

                Flux.mse(q1, target_batch)
            end
            Flux.update!(p.qnetwork1_state_tree, p.qnetwork1, q_grad_1[1])
            if p.use_popart
                update!(p.qnetwork1.layers[end], target_batch_popart)
            end

            q_grad_2 = Flux.gradient(p.qnetwork2) do qnetwork2
                q2 = dropdims(qnetwork2(q_input_batch), dims=1)

                ignore_derivatives() do
                    p.last_q2_mean = mean(q2)
                    p.last_critic2_loss = Flux.mse(q2, target_batch)
                end

                Flux.mse(q2, target_batch)
            end
            Flux.update!(p.qnetwork2_state_tree, p.qnetwork2, q_grad_2[1])
            if p.use_popart
                update!(p.qnetwork2.layers[end], target_batch_popart)
            end

            # polyak averaging
            for (dest, src) in zip(
                Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
                Flux.params([p.qnetwork1, p.qnetwork2]),
            )
                dest .= (1 - τ) .* dest .+ τ .* src
            end

            if p.use_popart
                # PopArt Polyak
                p.target_qnetwork1[end].μ = (1 - τ) .* p.target_qnetwork1[end].μ .+ τ .* p.qnetwork1[end].μ
                p.target_qnetwork1[end].σ = (1 - τ) .* p.target_qnetwork1[end].σ .+ τ .* p.qnetwork1[end].σ
                p.target_qnetwork2[end].μ = (1 - τ) .* p.target_qnetwork2[end].μ .+ τ .* p.qnetwork2[end].μ
                p.target_qnetwork2[end].σ = (1 - τ) .* p.target_qnetwork2[end].σ .+ τ .* p.qnetwork2[end].σ
            end

            if p.target_frac < 1.0f0
                μ_before_batch = to_device(collect(select_last_dim(μ_before_flatten_on_host, inds)))
                logσ_before_batch = to_device(collect(select_last_dim(logσ_before_flatten_on_host, inds)))
            end

            p_grad = Flux.gradient(p.actor) do actor
                aa, logp_π, μ, logσ = actor(p.device_rng, s_batch; is_sampling=true, is_return_log_prob=true, is_return_params=true)
                q_input = vcat(s_batch, aa)
                q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
                reward = mean(q)
                entropy = mean(logp_π)

                ignore_derivatives() do
                    p.last_reward_term = reward
                    p.last_entropy_term = α * entropy
                    p.last_actor_loss = α * entropy - reward
                end

                if p.target_frac < 1.0f0
                    # KL(new || old) for diagonal Gaussian policy, per-sample.
                    σ2   = exp.(2f0 .* logσ)
                    σ0_2 = exp.(2f0 .* logσ_before_batch)
                    t1 = (σ2 ./ σ0_2)
                    t2 = ((μ .- μ_before_batch).^2) ./ σ0_2
                    KLs = vec(sum(0.5f0 .* (t1 .+ t2 .- 1f0 .+ 2f0 .* (logσ_before_batch .- logσ)); dims=1))
                    KLs = Float32.(KLs)

                    fear_term = p.fear_factor * mean(KLs)
                else
                    fear_term = 0.0f0
                end


                actor_term = α * entropy - reward
                actor_term + fear_term
            end
            Flux.update!(p.actor_state_tree, p.actor, p_grad[1])
        end
    end


    if p.target_frac < 1.0f0
        μ_new, logσ_new = p.actor(p.device_rng, s)

        σ2   = exp.(2f0 .* logσ_new)
        σ0_2 = exp.(2f0 .* logσ_before)
        t1 = (σ2 ./ σ0_2)
        t2 = ((μ_new .- μ_before).^2) ./ σ0_2
        KLs = vec(sum(0.5f0 .* (t1 .+ t2 .- 1f0 .+ 2f0 .* (logσ_before .- logσ_new)); dims=1))
        KLs = Float32.(KLs)

        frac_changed = mean(KLs .> τ_change)
        ff_before = deepcopy(p.fear_factor)
        p.fear_factor = adjust_fear_factor(p.fear_factor, frac_changed; target=target_frac)

        if p.verbose
            #@show p.fear_factor - ff_before
            @show mean(logσ_before), mean(logσ_new), mean(μ_before), mean(μ_new)
            @show frac_changed
            @show p.fear_factor
        end
    end

    # Tune entropy automatically
    if p.automatic_entropy_tuning
        # p.log_α -= p.lr_alpha * p.α * mean(-logp_π′ .- p.target_entropy)

        log_α_grad = Flux.gradient(p.log_α) do log_α
            exp(log_α[1]) .* mean(-logp_π′ .- p.target_entropy)
        end

        α_now = exp(p.log_α[1])
        g_scaled = log_α_grad ./ max(α_now, 1e-12)
        Flux.update!(p.log_α_state_tree, p.log_α, g_scaled[1])

        clamp!(p.log_α, -12.5f0, 1.5f0)

        p.α = exp.(p.log_α)[1]

    end


    if p.verbose
        println("Reward Term = ",  p.last_reward_term)
        println("Entropy Term = ",  p.last_entropy_term)

        println("mean(-logπ′) = ",  mean(-logp_π′))
        println("target_entropy = ", p.target_entropy)
        println("alpha = ",  p.α)
    end

end



function _update!(p::SACPolicy2, t::AbstractTrajectory)
    #this is for imitation learning

    # if length(size(p.action_space)) == 2
    #     number_actuators = size(p.action_space)[2]
    # else
    #     #mono case 
    #     number_actuators = 1
    # end

    for i = 1:p.update_loops
        inds, batch = sample(p.rng, t, BatchSampler{SARTTS}(p.batch_size))
        update!(p, batch)
    end
end

function update_IL(p::SACPolicy2, trajectory::AbstractTrajectory)
    n_total = length(trajectory)
    n_total == 0 && return

    window = min(p.on_policy_update_freq, n_total)
    start_idx = rand(p.rng, 1:(n_total - window + 1))
    stop_idx = start_idx + window - 1

    temp_trajectory = Trajectory(
        state = trajectory[:state][:, :, start_idx:stop_idx],
        action = trajectory[:action][:, :, start_idx:stop_idx],
        reward = trajectory[:reward][:, start_idx:stop_idx],
        terminated = trajectory[:terminated][:, start_idx:stop_idx],
        truncated = trajectory[:truncated][:, start_idx:stop_idx],
        next_state = trajectory[:next_state][:, :, start_idx:stop_idx],
    )

    on_policy_update(p, temp_trajectory; whole_trajectory = true)
end



function update!(p::SACPolicy2, batch::NamedTuple{SARTTS})
    s, a, r, ter, trun, next_states = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α = p.γ, p.τ, p.α


    check_state_trees(p)


    acc_mu, acc_logp = antithetic_mean_sac2(p, next_states, α)

    next_values = acc_mu

    n_envs = size(ter, 1)
    next_values = reshape( next_values, n_envs, :)


    y = r .+ γ .* (1 .- ter) .* next_values

    
    logp_π′ = acc_logp

    p.last_target_q_mean = mean(y)
    p.last_mean_minus_log_pi = mean(-logp_π′)

    # Train Q Networks
    q_input = vcat(s, a)

    # println(repr(y))
    # error("abb")

    q_grad_1 = Flux.gradient(p.qnetwork1) do qnetwork1
        q1 = dropdims(qnetwork1(q_input), dims=1)

        ignore_derivatives() do
            p.last_q1_mean = mean(q1)
            p.last_critic1_loss = Flux.mse(q1, y)
        end

        Flux.mse(q1, y)
    end
    Flux.update!(p.qnetwork1_state_tree, p.qnetwork1, q_grad_1[1])
    if p.use_popart
        update!(p.qnetwork1.layers[end], y) 
    end

    q_grad_2 = Flux.gradient(p.qnetwork2) do qnetwork2
        q2 = dropdims(qnetwork2(q_input), dims=1)

        ignore_derivatives() do
            p.last_q2_mean = mean(q2)
            p.last_critic2_loss = Flux.mse(q2, y)
        end

        Flux.mse(q2, y)
    end
    Flux.update!(p.qnetwork2_state_tree, p.qnetwork2, q_grad_2[1])
    if p.use_popart
        update!(p.qnetwork2.layers[end], y) 
    end





    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end

    # Functors.fmap(p.target_qnetwork1, p.qnetwork1) do tgt, src
    #     if tgt isa AbstractArray && src isa AbstractArray
    #         @. tgt = (1 - τ) * tgt + τ * src
    #     end
    #     tgt
    # end

    # Functors.fmap(p.target_qnetwork2, p.qnetwork2) do tgt, src
    #     if tgt isa AbstractArray && src isa AbstractArray
    #         @. tgt = (1 - τ) * tgt + τ * src
    #     end
    #     tgt
    # end

    if p.use_popart
        # PopArt Polyak
        p.target_qnetwork1[end].μ = (1 - τ) .* p.target_qnetwork1[end].μ .+ τ .* p.qnetwork1[end].μ
        p.target_qnetwork1[end].σ = (1 - τ) .* p.target_qnetwork1[end].σ .+ τ .* p.qnetwork1[end].σ
        p.target_qnetwork2[end].μ = (1 - τ) .* p.target_qnetwork2[end].μ .+ τ .* p.qnetwork2[end].μ
        p.target_qnetwork2[end].σ = (1 - τ) .* p.target_qnetwork2[end].σ .+ τ .* p.qnetwork2[end].σ
    end

    p_grad = Flux.gradient(p.actor) do actor
        aa, logp_π = actor(p.device_rng, s; is_sampling=true, is_return_log_prob=true)
        q_input = vcat(s, aa)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        reward = mean(q)
        entropy = mean(logp_π)

        ignore_derivatives() do
            p.last_reward_term = reward
            p.last_entropy_term = α * entropy
            p.last_actor_loss = α * entropy - reward
        end

        α * entropy - reward
    end
    Flux.update!(p.actor_state_tree, p.actor, p_grad[1])

    if p.automatic_entropy_tuning
        log_α_grad = Flux.gradient(p.log_α) do log_α
            exp(log_α[1]) .* mean(-logp_π′ .- p.target_entropy)
        end

        α_now = exp(p.log_α[1])
        g_scaled = log_α_grad ./ max(α_now, 1f-12)
        Flux.update!(p.log_α_state_tree, p.log_α, g_scaled[1])

        clamp!(p.log_α, -12.5f0, 1.5f0)
        p.α = exp.(p.log_α)[1]
    end
end
