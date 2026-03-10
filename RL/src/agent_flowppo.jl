
Base.@kwdef mutable struct FlowPPOPolicy{A,D,R} <: AbstractPolicy
    approximator::A
    γ::Float32 = 0.99f0
    λ::Float32 = 0.99f0
    clip_range::Float32 = 0.2f0
    n_microbatches::Int = 1
    actorbatch_size = nothing
    n_epochs::Int = 5
    actor_loss_weight = 1.0f0
    critic_loss_weight = 0.5f0
    entropy_loss_weight = 0.01f0
    rng::R = Random.GLOBAL_RNG
    update_freq::Int = 2048
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
    λ_targets = 0.7f0
    n_targets = 100
    mm = ModulationModule()
    use_exploration_module::Bool = false

    antithetic_mean_samples = 4

    off_policy_update_freq = 0
    off_policy_batch_size = 256

    verbose = false

    last_action_log_prob::Vector{Float32} = [0.0f0]
    last_sigma::Vector{Float32} = [0.0f0]
    last_mu::Vector{Float32} = [0.0f0]
end




Base.@kwdef mutable struct FlowActorCritic{A,C}
    actor::A
    critic::C # value function approximator
    optimizer_actor = ADAM()
    optimizer_critic = ADAM()
    actor_state_tree = nothing
    critic_state_tree = nothing
end

@forward FlowActorCritic.actor device


# -------------------------
# Time embedding (sin/cos)
# -------------------------
const TWO_PI = 2f0 * Float32(pi)

"""
time_embed(t; L=16)

t: (1, B) Float32 matrix with values in [0,1]
returns: (2L, B) embedding
"""
function time_embed(t::AbstractArray{<:Real}; L::Int=16)
    t32 = Float32.(t)                      # (1,B)
    freqs = reshape(TWO_PI .* Float32.(1:L), L, 1)  # (L,1)
    ang = freqs .* t32                      # (L,B)
    return vcat(sin.(ang), cos.(ang))       # (2L,B)
end

# -------------------------
# Model: ctx encoder -> trunk -> token(k) -> head(k->1)
# vθ(x_t,t,ctx) returns scalar velocity (1,B)
# -------------------------
struct FlowMatchNet
    ctx_net
    trunk
    time_L::Int
end

Flux.@layer FlowMatchNet

function FlowMatchNet(n::Int, m::Int; ctx_dim::Int=64, hidden::Int=128, k::Int=1, time_L::Int=16)
    ctx_in = n + m
    ctx_net = Chain(
        Dense(ctx_in, ctx_dim, gelu),
        Dense(ctx_dim, ctx_dim, gelu),
    )
    # input: x_t (1) + t_emb (2*time_L) + ctx_emb (ctx_dim)
    in_dim = k + 2*time_L + ctx_dim
    trunk = Chain(
        Dense(in_dim, hidden, gelu),
        Dense(hidden, hidden, gelu),
        Dense(hidden, k),   
    )
    return FlowMatchNet(ctx_net, trunk, time_L)
end

"""
Forward:
x_t: (1,B)
t  : (1,B)
ctx: (n+m,B)
returns: v: (1,B)
"""
function (fm::FlowMatchNet)(x_t, t, ctx)
    t_emb  = time_embed(t; L=fm.time_L)     # (2L,B)
    c_emb  = fm.ctx_net(ctx)                # (ctx_dim,B)
    h_in   = vcat(x_t, t_emb, c_emb)        # (1+2L+ctx_dim, B)
    v  = fm.trunk(h_in)                 # (k,B)              # (1,B)
    return v
end

# -------------------------
# Flow-matching loss (Rectified Flow / straight path)
# x_t = (1-t)*x0 + t*x1, target u = x1 - x0 (constant)
# -------------------------
function fm_loss(fm::FlowMatchNet, s::AbstractArray, a::AbstractArray, y::AbstractArray; rng=Random.default_rng())
    B = size(s)

    ctx = vcat(s, a)                         # (n+m,B)
    x0  = randn(rng, Float32, 1, B[2:end]...)          # noise
    t   = rand(rng, Float32, 1, B[2:end]...)           # Uniform[0,1]
    x1  = Float32.(y)                        # data (1,B)
    x_t = (1 .- t) .* x0 .+ t .* x1
    u   = x1 .- x0
    v̂   = fm(x_t, t, ctx)
    return mean((v̂ .- u).^2)
end


# -------------------------
# Sampling (midpoint / RK2): more stable than Euler
# returns samples ŷ (1,B)
# -------------------------
function sample_midpoint(fm::FlowMatchNet, s::AbstractArray, a::AbstractArray;
                          steps::Int=30, rng=Random.default_rng(), x0=nothing)
    B = size(s)
    ctx = vcat(s, a)
    x = isnothing(x0) ? randn(rng, Float32, 1, B[2:end]...) : deepcopy(x0)
    dt = 1f0 / Float32(steps)

    for k in 1:steps
        t = Float32((k-1) / steps)
        tmat = fill(t, 1, B[2:end]...)
        v = fm(x, tmat, ctx)

        t_mid = t + 0.5f0 * dt
        tmidmat = fill(t_mid, 1, B[2:end]...)
        x_mid = x .+ (0.5f0 * dt) .* v
        v_mid = fm(x_mid, tmidmat, ctx)

        x .+= dt .* v_mid
    end
    return x
end




function create_agent_flow_ppo(;action_space, state_space, use_gpu, rng, y, p, update_freq = 2000, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, learning_rate_critic = nothing, fun = relu, fun_critic = nothing, tanh_end = false, n_envs = 1, clip1 = false, n_epochs = 10, n_microbatches = 1, actorbatch_size=nothing, normalize_advantage = true, logσ_is_network = true, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, noise = nothing, noise_scale = 90, dist = Normal, actor_update_freq = 1, λ_targets = 0.7f0, n_targets = 100, use_exploration_module = false, antithetic_mean_samples = 4, verbose = false, trajectory_size = 100_000, off_policy_update_freq = 0, off_policy_batch_size = 256, )

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

    mm = ModulationModule()

    if off_policy_update_freq == 0
        trajectory_size = update_freq
    end

    ctx_dim = Int(floor(10 * nna_scale_critic))
    hidden = Int(floor(20 * nna_scale_critic))
    critic = FlowMatchNet(ns, na; ctx_dim=ctx_dim, hidden=hidden, k=n_envs, time_L=16)
    critic = use_gpu ? critic |> gpu : critic



    approximator = isnothing(approximator) ? FlowActorCritic(
                actor = GaussianNetwork(
                    μ = create_chain(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, tanh_end = tanh_end),
                    logσ = create_logσ(logσ_is_network = logσ_is_network, ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, start_logσ = start_logσ),
                    logσ_is_network = logσ_is_network,
                    max_σ = max_σ
                ),
                critic = critic,
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
            ) : approximator

    Agent(
        policy = FlowPPOPolicy{typeof(approximator),dist,typeof(rng)}(
            approximator = approximator,
            γ = y,
            λ = p,
            clip_range = clip_range,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actorbatch_size = actorbatch_size,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            rng = rng,
            update_freq = update_freq,
            actor_update_freq = actor_update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            noise = noise,
            noise_sampler = noise_sampler,
            noise_scale = noise_scale,
            λ_targets = λ_targets,
            n_targets = n_targets,
            mm = mm,
            use_exploration_module = use_exploration_module,

            verbose = verbose,

            antithetic_mean_samples = antithetic_mean_samples,

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
    p::FlowPPOPolicy{<:FlowActorCritic{<:GaussianNetwork},Normal},
    state::AbstractArray,
    mask,
)
    μ, logσ = p.approximator.actor(send_to_device(device(p.approximator), state)) |> send_to_host

    StructArray{Normal}((μ, exp.(logσ)))
end

function prob(p::FlowPPOPolicy{<:FlowActorCritic,Categorical}, state::AbstractArray, mask)
    logits = p.approximator.actor(send_to_device(device(p.approximator), state))
    if !isnothing(mask)
        logits .+= ifelse.(mask, 0.0f0, typemin(Float32))
    end
    logits = logits |> softmax |> send_to_host
    
    [Categorical(x; check_args=false) for x in eachcol(logits)]
end

function prob(p::FlowPPOPolicy, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::FlowPPOPolicy, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::FlowPPOPolicy)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end



function (p::FlowPPOPolicy)(env::AbstractEnv)

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
    policy::FlowPPOPolicy,
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
end


function update!(
    trajectory::AbstractTrajectory,
    policy::FlowPPOPolicy,
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
    p::FlowPPOPolicy,
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

function update_IL(p::FlowPPOPolicy, trajectory::AbstractTrajectory)
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





function approximate_values(p::FlowPPOPolicy, states::AbstractArray)
    AC = p.approximator

    values = zeros(Float32, 1, size(states)[2:end]...)

    for i in 1:p.antithetic_mean_samples
        dist = prob(p, states, nothing)
        action = rand.(p.rng, dist)

        values += sample_midpoint(AC.critic, states, action; rng=p.rng)
    end

    values ./= p.antithetic_mean_samples

    values
end

function approximate_deltas(p::FlowPPOPolicy, states::AbstractArray, actions::AbstractArray)
    AC = p.approximator

    deltas = zeros(Float32, 1, size(states)[2:end]...)

    contexts = p.antithetic_mean_samples

    for i in 1:contexts
        x0 = randn(p.rng, Float32, 1, size(states)[2:end]...)

        v = sample_midpoint(AC.critic, states, actions; x0=x0)

        #temp_deltas = zeros(Float32, size(deltas)...)

        for j in 1:p.antithetic_mean_samples
            dist = prob(p, states, nothing)
            temp_actions = rand.(p.rng, dist)
            temp_v = sample_midpoint(AC.critic, states, temp_actions; x0=x0)

            temp_deltas = v .- temp_v

            indices = findall(i -> abs(temp_deltas[i]) > abs(deltas[i]), eachindex(temp_deltas))

            deltas[indices] = temp_deltas[indices]

            # @show v, temp_v, temp_deltas
            # error("stop")
        end
        #temp_deltas ./= p.antithetic_mean_samples

        #deltas .+= temp_deltas
    end

    #deltas ./= p.antithetic_mean_samples

    deltas
end


function check_state_trees(p::FlowPPOPolicy)
    # just to make sure the state trees are initialized

    AC = p.approximator

    reset_optimizers = false #(p.update_step / p.update_freq) % 400 == 0
    start_optimizers = isnothing(AC.actor_state_tree) || isnothing(AC.critic_state_tree)

    if start_optimizers || reset_optimizers
        if p.verbose
            println("________________________________________________________________________")
            println("Reset Optimizers")
            println("________________________________________________________________________")
        end
        AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor)
        AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)

        #Optimisers.adjust!(AC.critic_state_tree.layers[end]; lambda = 0.0)
    end

    return
end


function _update!(p::FlowPPOPolicy, t::Any; update_actor = true, update_critic = true, microbatch_size = nothing)


    rng = p.rng
    AC = p.approximator
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight
    D = RL.device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_trajectory = size(t[:terminated])
    n_rollout = min(p.update_freq, n_trajectory)


    
    global valid_indices = collect(n_trajectory-n_rollout+1:n_trajectory)


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


    rewards = collect(to_device(t[:reward][:,valid_indices]))
    terminated = collect(to_device(t[:terminated][:,valid_indices]))
    truncated = collect(to_device(t[:truncated][:,valid_indices]))


    gae_deltas = approximate_deltas(p, states, actions)
    gae_deltas = reshape(gae_deltas, n_envs, :)

    advantages, _ = generalized_advantage_estimation(
        gae_deltas,
        zeros(Float32, size(gae_deltas)),
        zeros(Float32, size(gae_deltas)),
        γ,
        λ;
        dims=2,
        terminated=terminated,
        truncated=truncated,
    )

    advantages = to_device(advantages)



    actions_flatten = flatten_batch(t[:action][:,:,valid_indices])
    action_log_probs = to_device(t[:action_log_prob][:,valid_indices])
    explore_mod = to_device(t[:explore_mod][:,valid_indices])

    stop_update = false

    actor_losses = Float32[]
    critic_losses = Float32[]
    critic2_losses = Float32[]
    entropy_losses = Float32[]


    check_state_trees(p)



    next_states = to_device(flatten_batch(t[:next_state][:,:,valid_indices]))
    next_values = approximate_values(p, next_states)
    targets = td_lambda_targets(rewards, terminated, truncated, next_values, γ; λ = p.λ_targets)
    

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, collect(1:n_samples))

        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            if isnothing(actorbatch_size)
                inds_actor = inds
            else
                inds_actor = inds[1:clamp(actorbatch_size, 1, length(inds))]
            end


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

            tar = targets[:,inds]

            
            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-8 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            # s_neg = sample_negatives_far(s)

            g_actor, g_critic = Flux.gradient(AC.actor, AC.critic) do actor, critic

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
                
                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))

                critic_loss = fm_loss(critic, s, a, tar; rng=p.rng)


                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss 


                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)

                end

                loss
            end
            
            if !stop_update
                if (p.update_step / p.update_freq) % p.actor_update_freq == 0
                    if update_actor
                        Flux.update!(AC.actor_state_tree, AC.actor, g_actor)
                    end
                end
                if update_critic
                    Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
                end
            else
                break
            end


        end

        if stop_update
            break
        end
    end

    
    mean_actor_loss = mean(abs.(actor_losses))
    mean_critic_loss = mean(abs.(critic_losses))
    mean_critic2_loss = mean(abs.(critic2_losses))
    mean_entropy_loss = mean(abs.(entropy_losses))
    # mean_logσ_regularization_loss = mean(abs.(logσ_regularization_losses))
    # mean_critic_regularization_loss = mean(abs.(critic_regularization_losses))
    
    if p.verbose
        println("---")
        println("mean actor loss: $(mean_actor_loss)")
        println("mean critic loss: $(mean_critic_loss)")
        println("mean critic2 loss: $(mean_critic2_loss)")
        println("mean entropy loss: $(mean_entropy_loss)")
        println("---")
    end
end


function _update_off_policy!(p::FlowPPOPolicy, batch;)

    check_state_trees(p)

    AC = p.approximator
    γ = p.γ
    w₂ = p.critic_loss_weight

    D = RL.device(AC)
    s, a, r, t, next_states = send_to_device(D, batch)

    n_envs = size(t)[1]

    next_values = approximate_values(p, flatten_batch(next_states))

    a = flatten_batch(a)
    s = flatten_batch(s)

    targets_critic = r + next_values .* γ .* (1 .- t)


    g_critic = Flux.gradient(AC.critic) do critic

        critic_loss = fm_loss(critic, s, a, targets_critic; rng=p.rng)


        loss = w₂ * critic_loss

        loss
    end
    
    Flux.update!(AC.critic_state_tree, AC.critic, g_critic[1])

end
