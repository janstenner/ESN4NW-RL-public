function create_NNA(;na, ns, use_gpu, is_actor, init, copyfrom = nothing, nna_scale, network_depth = 2, drop_middle_layer = nothing, learning_rate = 0.001, fun = relu)
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
        push!(layers, Dense(nna_size_actor, na, tanh; init = init))
        n = Chain(layers...)
    else
        layers = Any[Dense(ns + na, nna_size_critic, fun; init = init)]
        for _ in 2:network_depth
            push!(layers, Dense(nna_size_critic, nna_size_critic, fun; init = init))
        end
        push!(layers, Dense(nna_size_critic, 1; init = init))
        n = Chain(layers...)
    end

    # nna = CustomNeuralNetworkApproximator(
    #     model = use_gpu ? n |> gpu : n,
    #     optimizer = Flux.Adam(learning_rate),
    # )

    nna = use_gpu ? n |> gpu : n

    if copyfrom !== nothing
        copyto!(nna, copyfrom) 
    end

    nna
end


function create_agent(;action_space, state_space, use_gpu, rng, y, p, batch_size,
                    start_steps, start_policy, update_after, update_freq, update_loops = 1, reset_stage = POST_EPISODE_STAGE, act_limit,
                    act_noise, noise_hold = 1,
                    nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, memory_size = 0, trajectory_length = 1000, mono = false, learning_rate = 0.001, learning_rate_critic = nothing, fun = relu, fun_critic = nothing, clip_grad = 0.5, betas = (0.9, 0.999), verbose = false)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    !isnothing(drop_middle_layer)        &&  (network_depth = drop_middle_layer ? 1 : 2)
    !isnothing(drop_middle_layer_critic) &&  (network_depth_critic = drop_middle_layer_critic ? 1 : 2)
    isnothing(network_depth_critic)      &&  (network_depth_critic = network_depth)
    network_depth = max(1, Int(network_depth))
    network_depth_critic = max(1, Int(network_depth_critic))
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(learning_rate_critic)     &&  (learning_rate_critic = learning_rate)
    
    init = Flux.glorot_uniform(rng)
    
    behavior_actor = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, learning_rate = learning_rate, fun = fun)

    behavior_critic = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, learning_rate = learning_rate_critic, fun = fun_critic)


    target_actor = deepcopy(behavior_actor)
    target_critic = deepcopy(behavior_critic)

    if mono
        reward_size = 1
    else
        reward_size = size(action_space)[1]
    end

    actions = zeros(size(action_space))
    last_noise = randn(rng, size(actions[1:end-memory_size,:])) .* act_noise
    
    Agent(
        policy = CustomDDPGPolicy(
            action_space = action_space,
            state_space = state_space,

            rng = rng,

            target_actor = target_actor,
            target_critic = target_critic,
            behavior_actor = behavior_actor,
            behavior_critic = behavior_critic,

            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
            optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas)),

            use_gpu = use_gpu,

            y = y,
            p = p,
            batch_size = batch_size,
            start_steps = start_steps,
            start_policy = start_policy,
            update_after = update_after,
            update_freq = update_freq,
            update_loops = update_loops,
            reset_stage = reset_stage,
            act_limit = act_limit,
            act_noise = act_noise,
            noise_hold = noise_hold,
            last_noise = last_noise,
            memory_size = memory_size,
            verbose = verbose,
        ),
        trajectory = 
            CircularArrayTrajectory(;
                capacity = trajectory_length,
                state = Float32 => size(state_space)[1],
                action = Float32 => size(action_space)[1],
                reward = Float32 => reward_size,
                terminated = Bool => (),
                truncated = Bool => ()
            ),
    )
end

Base.@kwdef mutable struct CustomDDPGPolicy{
    P,
    R,
} <: AbstractPolicy

    action_space::Space
    state_space::Space

    rng::R

    behavior_actor
    behavior_critic
    target_actor
    target_critic

    optimizer_actor = ADAM()
    optimizer_critic = ADAM()
    actor_state_tree = nothing
    critic_state_tree = nothing

    use_gpu::Bool

    y
    p
    batch_size
    start_steps
    start_policy::P
    update_after
    update_freq
    update_loops
    reset_stage
    act_limit
    act_noise
    noise_hold
    last_noise
    memory_size
    verbose::Bool = false

    update_step::Int = 0
    actor_loss::Float32 = 0.0f0
    critic_loss::Float32 = 0.0f0
end


# Flux.functor(x::CustomDDPGPolicy) = (
#     ba = x.behavior_actor,
#     bc = x.behavior_critic,
#     ta = x.target_actor,
#     tc = x.target_critic,
# ),
# y -> begin
#     x = @set x.behavior_actor = y.ba
#     x = @set x.behavior_critic = y.bc
#     x = @set x.target_actor = y.ta
#     x = @set x.target_critic = y.tc
#     x
# end


function (policy::CustomDDPGPolicy)(env; learning = true, test = false)

    if policy.update_step <= policy.start_steps
        policy.start_policy(env)
    else
        D = device(policy.behavior_actor)
        s = state(env)

        s = send_to_device(D, s)
        
        if test == false
            actions = policy.behavior_actor(s)
        else
            actions = zeros(size(policy.action_space))

            for i in 1:length(actions)
                actions[i] = policy.behavior_actor(s[:,i])[1]
            end
        end

        actions = actions |> send_to_host

        if learning
            if policy.update_step % policy.noise_hold == 0
                policy.last_noise = randn(policy.rng, size(actions[1:end-policy.memory_size,:])) .* policy.act_noise
            end
            actions[1:end-policy.memory_size,:] += policy.last_noise
            actions = clamp.(actions, -policy.act_limit, policy.act_limit)
        else
            actions = clamp.(actions, -policy.act_limit, policy.act_limit)
        end

        actions
    end
end



function (policy::CustomDDPGPolicy)(stage::AbstractStage, env::AbstractEnv)
    nothing
end

function update!(
    policy::CustomDDPGPolicy,
    traj::Trajectory,
    ::AbstractEnv,
    stage::PostEpisodeStage,
)
    if stage == policy.reset_stage
        policy.update_step = 0
    end
end

function update!(
    policy::CustomDDPGPolicy,
    traj::Trajectory,
    ::AbstractEnv,
    stage::PostExperimentStage,
)
    if stage == policy.reset_stage
        policy.update_step = 0
    end
end

function update!(
    trajectory::AbstractTrajectory,
    p::CustomDDPGPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    
end

function update!(
    trajectory::AbstractTrajectory,
    policy::CustomDDPGPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    policy.update_step += 1

    s = state(env)

    #We assume that action_space[2] == state_space[2]
    for i in 1:size(s)[2]
        push!(trajectory[:state], s[:,i])
        push!(trajectory[:action], action[:,i])
    end
end

function update!(
    trajectory::AbstractTrajectory,
    policy::CustomDDPGPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)

    #We assume that state_space[2] == size(r)[1]
    done = is_terminated(env) || is_truncated(env)
    trunc = is_truncated(env)
    for i in 1:size(r)[1]
        push!(trajectory[:reward], r[i])
        # SARTS samplers consume :terminated as the done-mask in this agent.
        push!(trajectory[:terminated], done)
        push!(trajectory[:truncated], trunc)
    end
end

function update!(
    trajectory::AbstractTrajectory,
    policy::CustomDDPGPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    
end


function update!(
    policy::CustomDDPGPolicy,
    traj::Trajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    if length(size(policy.action_space)) == 2
        number_actuators = size(policy.action_space)[2]
    else
        #mono case 
        number_actuators = 1
    end
    length(traj) > policy.update_after * number_actuators || return
    policy.update_step % policy.update_freq == 0 || return
    #println("UPDATE!")
    for i = 1:policy.update_loops
        inds, batch = pde_sample(policy.rng, traj, BatchSampler{SARTS}(policy.batch_size), number_actuators)
        update!(policy, batch)
    end
end

function _update!(p::CustomDDPGPolicy, t::AbstractTrajectory)
    #this is for imitation learning

    if length(size(p.action_space)) == 2
        number_actuators = size(p.action_space)[2]
    else
        #mono case 
        number_actuators = 1
    end

    for i = 1:p.update_loops
        inds, batch = pde_sample(p.rng, t, BatchSampler{SARTS}(p.batch_size), number_actuators)
        update!(p, batch)
    end
end

update_IL(p::CustomDDPGPolicy, t::AbstractTrajectory) = _update!(p, t)

function update!(policy::CustomDDPGPolicy, batch::NamedTuple{SARTS})
    
    s, a, r, terminated, snext = batch

    if isnothing(policy.actor_state_tree) || isnothing(policy.critic_state_tree)
        if policy.verbose
            println("________________________________________________________________________")
            println("Reset Optimizers")
            println("________________________________________________________________________")
        end
        policy.actor_state_tree = Flux.setup(policy.optimizer_actor, policy.behavior_actor)
        policy.critic_state_tree = Flux.setup(policy.optimizer_critic, policy.behavior_critic)
    end

    A = policy.behavior_actor
    C = policy.behavior_critic
    Aₜ = policy.target_actor
    Cₜ = policy.target_critic

    y = policy.y
    p = policy.p

    # r = vec(r)'
    # new_t = Array{Float32}(undef, 1, 0)
    # for i in 1:length(t)
    #     new_t = hcat(new_t, t[i] * ones(size(policy.action_space)[2])')
    # end
    
    # s, a, r, terminated, snext = send_to_device(device(policy), (s, a, r, new_t, snext))

    s, a, r, terminated, snext = send_to_device(device(policy), (s, a, r, terminated, snext))

    anext = Aₜ(snext)
    qₜ = Cₜ(vcat(snext, anext)) |> vec
    #qₜ = - abs.( (s .* 50 - 30 .* Aₜ(snext)) / 50 ) |> vec
    qnext = r .+ y .* (1 .- terminated) .* qₜ
    a = Flux.unsqueeze(a, ndims(a)+1)

    critic_grad = Flux.gradient(C) do critic
        q = critic(vcat(s, a)) |> vec
        loss = mean((qnext .- q) .^ 2)
        ignore() do
            policy.critic_loss = loss
        end
        loss
    end
    Flux.update!(policy.critic_state_tree, C, critic_grad[1])

    actor_grad = Flux.gradient(A) do actor
        loss = -mean(C(vcat(s, actor(s))))
        #loss = abs.( (s .* 500 - 30 .* A(s)) / 500 )[1]
        ignore() do
            policy.actor_loss = loss
        end
        loss
    end
    Flux.update!(policy.actor_state_tree, A, actor_grad[1])


    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= p .* dest .+ (1 - p) .* src
    end
end


