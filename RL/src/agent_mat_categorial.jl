function _to_3d(x)
    ndims(x) == 2 ? reshape(x, size(x, 1), size(x, 2), 1) : x
end

function _mat_cat_decode_logits(encoder, decoder, state, n_actors::Int, actions_forced::Union{Nothing,AbstractMatrix{<:Integer}} = nothing)
    D = device(encoder)
    s = _to_3d(send_to_device(D, state))

    obs_rep, v = encoder(s)
    na = size(decoder.embedding.weight, 2)
    B = size(obs_rep, 3)

    prefix = send_to_device(D, zeros(Float32, na, 1, B))
    logits = nothing

    for n in 1:n_actors
        logits_seq, _ = decoder(prefix, obs_rep[:, 1:n, :])
        logits_n = logits_seq[:, end:end, :]
        logits = isnothing(logits) ? logits_n : cat(logits, logits_n; dims=2)

        if n < n_actors
            if isnothing(actions_forced)
                logits_host = send_to_host(reshape(logits_n, na, B))
                sampled = Vector{Int}(undef, B)
                for b in 1:B
                    sampled[b] = argmax(softmax(logits_host[:, b]))
                end
                tok = Float32.(Flux.onehotbatch(sampled, 1:na))
            else
                tok = Float32.(Flux.onehotbatch(vec(actions_forced[n, :]), 1:na))
            end
            tok3 = reshape(tok, na, 1, B)
            prefix = cat(prefix, send_to_device(D, tok3), dims=2)
        end
    end

    @assert !isnothing(logits) "n_actors must be >= 1"
    logits, v
end

function _mat_cat_sample_action_and_logprob(p, state; deterministic::Bool = false)
    D = device(p.encoder)
    s = _to_3d(send_to_device(D, state))

    obs_rep, _ = p.encoder(s)
    na = size(p.decoder.embedding.weight, 2)
    B = size(obs_rep, 3)
    @assert B == 1 "AbstractEnv path expects batch size 1."

    prefix = send_to_device(D, zeros(Float32, na, 1, 1))
    action = Vector{Int}(undef, p.n_actors)
    log_p = Vector{Float32}(undef, p.n_actors)

    for n in 1:p.n_actors
        logits_seq, _ = p.decoder(prefix, obs_rep[:, 1:n, :])
        logits_n = reshape(send_to_host(logits_seq[:, end:end, :]), na)
        probs_n = softmax(logits_n)

        a_n = deterministic ? argmax(probs_n) : rand(p.rng, Categorical(vec(probs_n); check_args=false))
        action[n] = a_n
        log_p[n] = Float32(log(probs_n[a_n]))

        if n < p.n_actors
            tok = Float32.(Flux.onehotbatch([a_n], 1:na))
            tok3 = reshape(tok, na, 1, 1)
            prefix = cat(prefix, send_to_device(D, tok3), dims=2)
        end
    end

    action, log_p
end

function create_agent_mat_categorial(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = leakyrelu, fun_critic = nothing, n_actors = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = true, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, adaptive_weights = false, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, dim_model = 64, block_num = 1, head_num = 4, head_dim = nothing, ffn_dim = 120, drop_out = 0.1, betas = (0.99, 0.99), jointPPO = false, customCrossAttention = true, one_by_one_training = false, clip_range = 0.2f0, tanh_end = false, positional_encoding = 1, useSeparateValueChain = false, verbose = false)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    !isnothing(drop_middle_layer)        &&  (network_depth = drop_middle_layer ? 1 : 2)
    !isnothing(drop_middle_layer_critic) &&  (network_depth_critic = drop_middle_layer_critic ? 1 : 2)
    isnothing(network_depth_critic)      &&  (network_depth_critic = network_depth)
    network_depth = max(1, Int(network_depth))
    network_depth_critic = max(1, Int(network_depth_critic))
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(head_dim)                 &&  (head_dim = Int(floor(dim_model / head_num)))

    init = Flux.glorot_uniform(rng)

    ns = size(state_space)[1]
    na = size(action_space)[1]

    context_size = n_actors

    if jointPPO
        encoder_input_dim = dim_model * n_actors
    else
        encoder_input_dim = dim_model
    end

    if network_depth_critic == 1
        head_encoder = Dense(encoder_input_dim, 1)
    else
        encoder_layers = Any[Dense(encoder_input_dim, ffn_dim, fun)]
        for _ in 3:network_depth_critic
            push!(encoder_layers, Dense(ffn_dim, ffn_dim, fun))
        end
        push!(encoder_layers, Dense(ffn_dim, 1))
        head_encoder = Chain(encoder_layers...)
    end

    if network_depth == 1
        # For categorical policy the decoder head must emit logits (no tanh).
        head_decoder = Dense(dim_model, na)
        head_decoder.weight[:] *= 0.01
        head_decoder.bias[:] *= 0.01
    else
        decoder_layers = Any[Dense(dim_model, ffn_dim, fun)]
        for _ in 3:network_depth
            push!(decoder_layers, Dense(ffn_dim, ffn_dim, fun))
        end
        push!(decoder_layers, Dense(ffn_dim, na))
        head_decoder = Chain(decoder_layers...)
    end

    decoder_blocks = BlockStack(block_num, dim_model, head_num, head_dim, ffn_dim; pdrop=drop_out, useCustomCrossAttention=customCrossAttention)
    encoder_blocks = Chain(ntuple(_ -> MATEncoderBlock(dim_model, head_num, head_dim, ffn_dim; pdrop=drop_out), block_num)...)

    if positional_encoding == 1
        position_encoding_encoder = SinCosPositionEmbed(dim_model)
        position_encoding_decoder = SinCosPositionEmbed(dim_model)
    elseif positional_encoding == 2
        position_encoding_encoder = Embedding(context_size => dim_model)
        position_encoding_decoder = Embedding(context_size => dim_model)
    elseif positional_encoding == 3
        position_encoding_encoder = ZeroEncoding(dim_model)
        position_encoding_decoder = ZeroEncoding(dim_model)
    else
        error("Unknown positional_encoding=$(positional_encoding)")
    end

    if useSeparateValueChain
        embedding_v = Dense(ns, dim_model, bias=false)
        position_encoding_v = deepcopy(position_encoding_encoder)
        ln_v = LayerNorm(dim_model)
        dropout_v = Dropout(drop_out)
        encoder_blocks_v = Chain(ntuple(_ -> MATEncoderBlock(dim_model, head_num, head_dim, ffn_dim; pdrop=drop_out), block_num)...)
    else
        embedding_v = nothing
        position_encoding_v = nothing
        ln_v = nothing
        dropout_v = nothing
        encoder_blocks_v = nothing
    end

    encoder = MATEncoder(
        embedding = Dense(ns, dim_model, bias=false),
        position_encoding = position_encoding_encoder,
        ln = LayerNorm(dim_model),
        dropout = Dropout(drop_out),
        blocks = encoder_blocks,

        embedding_v = embedding_v,
        position_encoding_v = position_encoding_v,
        ln_v = ln_v,
        dropout_v = dropout_v,
        blocks_v = encoder_blocks_v,

        head = head_encoder,

        jointPPO = jointPPO,
        useSeparateValueChain = useSeparateValueChain,
    )

    decoder = MATDecoder(
        embedding = Dense(na, dim_model, bias=false),
        position_encoding = position_encoding_decoder,
        ln = LayerNorm(dim_model),
        dropout = Dropout(drop_out),
        blocks = decoder_blocks,
        head = head_decoder,
        # Keep logσ for structural compatibility, but not used in categorical path.
        logσ = create_logσ_mat(logσ_is_network = logσ_is_network, ns = dim_model, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, start_logσ = start_logσ),
        logσ_is_network = logσ_is_network,
        max_σ = max_σ,
    )

    encoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas))
    decoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas))

    encoder_state_tree = Flux.setup(encoder_optimizer, encoder)
    decoder_state_tree = Flux.setup(decoder_optimizer, decoder)

    Agent(
        policy = MATCategorialPolicy(
            encoder = encoder,
            decoder = decoder,
            encoder_optimizer = encoder_optimizer,
            decoder_optimizer = decoder_optimizer,
            encoder_state_tree = encoder_state_tree,
            decoder_state_tree = decoder_state_tree,
            n_actors = n_actors,
            γ = y,
            λ = p,
            clip_range = clip_range,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            adaptive_weights = adaptive_weights,
            rng = rng,
            update_freq = update_freq,
            clip1 = clip1,
            normalize_advantage = normalize_advantage,
            start_steps = start_steps,
            start_policy = start_policy,
            target_kl = target_kl,
            jointPPO = jointPPO,
            one_by_one_training = one_by_one_training,
            verbose = verbose,
        ),
        trajectory = CircularArrayTrajectory(
            capacity = update_freq,
            state = Float32 => (size(state_space)[1], n_actors),
            action = Int => (1, n_actors),
            action_log_prob = Float32 => (n_actors),
            reward = Float32 => (n_actors),
            terminated = Bool => (n_actors,),
            truncated = Bool => (n_actors,),
            next_state = Float32 => (size(state_space)[1], n_actors),
        ),
    )
end

Base.@kwdef mutable struct MATCategorialPolicy <: AbstractPolicy
    encoder
    decoder
    encoder_optimizer
    decoder_optimizer
    encoder_state_tree
    decoder_state_tree
    n_actors::Int = 0
    γ::Float32 = 0.99f0
    λ::Float32 = 0.95f0
    clip_range::Float32 = 0.2f0
    n_microbatches::Int = 4
    n_epochs::Int = 4
    actor_loss_weight::Float32 = 1.0f0
    critic_loss_weight::Float32 = 0.5f0
    entropy_loss_weight::Float32 = 0.0f0
    adaptive_weights::Bool = false
    rng = StableRNG()
    update_freq::Int = 200
    update_step::Int = 0
    clip1::Bool = false
    normalize_advantage::Bool = true
    start_steps = -1
    start_policy = nothing
    target_kl = 100.0
    verbose::Bool = false
    last_action_log_prob::Vector{Float32} = [0.0]
    next_values::Vector{Float32} = [0.0]
    jointPPO::Bool = false
    one_by_one_training::Bool = false
end

function prob(p::MATCategorialPolicy, state::AbstractArray, mask)
    logits, _ = _mat_cat_decode_logits(p.encoder, p.decoder, state, p.n_actors, nothing)
    logits = send_to_host(logits)

    B = size(logits, 3)
    @assert B == 1 "prob(::MATCategorialPolicy, state) currently supports batch size 1."

    [
        Categorical(vec(softmax(logits[:, n, 1])); check_args=false) for n in 1:p.n_actors
    ]
end

function prob(p::MATCategorialPolicy, env::AbstractEnv)
    prob(p, state(env), nothing)
end

function (p::MATCategorialPolicy)(env::AbstractEnv)
    if p.update_step <= p.start_steps
        action_raw = p.start_policy(env)
        action = vec(Int.(action_raw))
        if length(action) == 1 && p.n_actors > 1
            action = fill(action[1], p.n_actors)
        end
        @assert length(action) == p.n_actors "start_policy must return $(p.n_actors) discrete actions."

        logits, _ = _mat_cat_decode_logits(p.encoder, p.decoder, state(env), p.n_actors, reshape(action, p.n_actors, 1))
        log_p = logsoftmax(logits; dims=1)
        na = size(logits, 1)
        one_hot = Float32.(reshape(Flux.onehotbatch(action, 1:na), na, p.n_actors, 1))
        one_hot = send_to_device(device(p.encoder), one_hot)
        log_p_a = vec(send_to_host(sum(log_p .* one_hot; dims=1)))
        p.last_action_log_prob = Float32.(log_p_a)
        reshape(action, 1, p.n_actors)
    else
        action, log_p = _mat_cat_sample_action_and_logprob(p, state(env); deterministic=false)
        p.last_action_log_prob = Float32.(log_p)
        reshape(action, 1, p.n_actors)
    end
end

function (p::MATCategorialPolicy)(env::MultiThreadEnv)
    @error "MATCategorialPolicy for MultiThreadEnv is not implemented yet."
end

function update!(
    trajectory::AbstractTrajectory,
    policy::MATCategorialPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    action_vec = vec(Int.(action))
    if length(action_vec) == 1 && policy.n_actors > 1
        action_vec = fill(action_vec[1], policy.n_actors)
    end

    push!(
        trajectory;
        state = state(env),
        action = reshape(action_vec, 1, policy.n_actors),
        action_log_prob = policy.last_action_log_prob,
    )
end

function update!(
    trajectory::AbstractTrajectory,
    policy::MATCategorialPolicy,
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
    p::MATCategorialPolicy,
    t::AbstractTrajectory,
    ::AbstractEnv,
    ::PostActStage,
)
    length(t) == 0 && return
    p.update_step += 1
    if p.update_step % p.update_freq == 0
        _update!(p, t)
    end
end

function update_IL(p::MATCategorialPolicy, trajectory::AbstractTrajectory)
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
        terminated = trajectory[:terminated][:, start_idx:stop_idx],
        truncated = trajectory[:truncated][:, start_idx:stop_idx],
        next_state = trajectory[:next_state][:, :, start_idx:stop_idx],
    )

    _update!(p, temp_trajectory)
end

function _update!(p::MATCategorialPolicy, t::Any)
    rng = p.rng
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight

    D = device(p.encoder)
    to_device(x) = send_to_device(D, x)

    n_actors, n_rollout = size(t[:terminated])

    @assert n_rollout % n_microbatches == 0 "size mismatch"

    microbatch_size = Int(floor(n_rollout ÷ n_microbatches))

    n = length(t)
    states = to_device(t[:state])
    next_states = to_device(t[:next_state])

    _, values = p.encoder(states)
    _, next_values = p.encoder(next_states)

    values = reshape(send_to_host(values), n_actors, :)
    next_values = reshape(send_to_host(next_values), n_actors, :)

    rewards = t[:reward]
    terminated = t[:terminated]
    truncated = t[:truncated]

    if p.jointPPO
        values = values[1, :]
        next_values = next_values[1, :]
        rewards = rewards[1, :]
        terminated = terminated[1, :]
        truncated = truncated[1, :]
    end

    advantages, _ = generalized_advantage_estimation(
        rewards,
        values,
        next_values,
        γ,
        λ;
        dims=2,
        terminated=terminated,
        truncated=truncated,
    )

    if p.jointPPO
        returns = to_device(advantages .+ values[1:n_rollout])
    else
        returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    end
    advantages = to_device(advantages)

    action_log_probs = t[:action_log_prob]
    if p.jointPPO
        action_log_probs = sum(action_log_probs, dims=1)[:]
    else
        action_log_probs = reshape(action_log_probs, 1, size(action_log_probs, 1), size(action_log_probs, 2))
    end

    stop_update = false

    if p.one_by_one_training
        n_epochs = n_epochs * p.n_actors
    end

    actor_losses = Float32[]
    critic_losses = Float32[]

    for epoch in 1:n_epochs
        rand_inds = shuffle!(rng, Vector(1:n_rollout))

        for i in 1:n_microbatches
            inds = rand_inds[(i - 1) * microbatch_size + 1:i * microbatch_size]

            s = to_device(collect(select_last_dim(states, inds)))
            a_int = Int.(reshape(collect(select_last_dim(t[:action], inds)), p.n_actors, :))

            r = select_last_dim(returns, inds)
            log_p = select_last_dim(action_log_probs, inds)
            adv = select_last_dim(advantages, inds)

            clamp!(log_p, log(1e-8), Inf)

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            g_encoder, g_decoder = Flux.gradient(p.encoder, p.decoder) do encoder, decoder
                logits, v′ = _mat_cat_decode_logits(encoder, decoder, s, p.n_actors, a_int)

                na = size(logits, 1)
                B = size(logits, 3)
                log_p′ = logsoftmax(logits; dims=1)
                p′ = softmax(logits; dims=1)

                one_hot = Float32.(reshape(Flux.onehotbatch(vec(a_int), 1:na), na, p.n_actors, B))
                one_hot = send_to_device(device(encoder), one_hot)
                log_p′ₐ = sum(log_p′ .* one_hot; dims=1)

                if p.jointPPO
                    log_p′ₐ = sum(log_p′ₐ, dims=2)[:]
                end

                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host
                    if approx_kl_div > p.target_kl && (i > 1 || epoch > 1)
                        if p.verbose
                            println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        end
                        stop_update = true
                    end
                end

                entropy_values = -sum(p′ .* log_p′; dims=1)

                if p.jointPPO
                    entropy_loss = mean(entropy_values)

                    surr1 = ratio .* adv
                    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))
                    critic_loss = mean((r .- v′[1, 1, :]) .^ 2)
                elseif p.one_by_one_training
                    actor_index = 1 + epoch % p.n_actors

                    entropy_loss = mean(entropy_values[1, actor_index, :])

                    surr1 = ratio[1, actor_index, :] .* adv[actor_index, :]
                    surr2 = clamp.(ratio[1, actor_index, :], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv[actor_index, :]

                    actor_loss = -mean(min.(surr1, surr2))
                    critic_loss = mean((r[actor_index, :] .- v′[1, actor_index, :]) .^ 2)
                else
                    entropy_loss = mean(entropy_values)

                    surr1 = ratio[1, :, :] .* adv
                    surr2 = clamp.(ratio[1, :, :], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))
                    critic_loss = mean((r .- v′[1, :, :]) .^ 2)
                end

                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

                ignore() do
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                end

                loss
            end

            if !stop_update
                Flux.update!(p.encoder_state_tree, p.encoder, g_encoder)
                Flux.update!(p.decoder_state_tree, p.decoder, g_decoder)
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

    if p.verbose
        println("---")
        println(mean_actor_loss)
        println(mean_critic_loss)
    end

    if p.adaptive_weights
        actor_factor = clamp(0.5 / mean_actor_loss, 0.9, 1.1)
        critic_factor = clamp(0.3 / mean_critic_loss, 0.9, 1.1)

        if p.verbose
            println("changing actor weight from $(w₁) to $(w₁ * actor_factor)")
            println("changing critic weight from $(w₂) to $(w₂ * critic_factor)")
        end

        p.actor_loss_weight = w₁ * actor_factor
        p.critic_loss_weight = w₂ * critic_factor
    end

    if p.verbose
        println("---")
    end
end
