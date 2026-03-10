
struct MATEncoderBlock{MHA,LN1,LN2,FF,DO}
    mha::MHA
    ln1::LN1
    ln2::LN2
    ff::FF
    dropout::DO
end

function MATEncoderBlock(d_model::Int, nheads::Int, head_dim::Int, d_ff::Int; pdrop=0.1)
    MATEncoderBlock(
        MultiHeadAttention(d_model => head_dim*nheads => d_model, nheads=nheads, dropout_prob=pdrop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Chain(Dense(d_model, d_ff, gelu), Dense(d_ff, d_model)),
        Dropout(pdrop)
    )
end

Flux.@layer MATEncoderBlock trainable=(mha, ln1, ln2, ff)

function (m::MATEncoderBlock)(x)
    # x: (d_model, T, B)
    y, _ = m.mha(x, x, x; mask=nothing)   # Self-Attention
    # x = m.ln1(x .+ m.dropout(y))
    x = x .+ m.dropout(y)
    z = m.ff(x)
    # x = m.ln2(x .+ m.dropout(z))
    x = x .+ m.dropout(z)
    return x
end




struct MATDecoderBlock{MHA1,MHA2,LN1,LN2,LN3,FF,DO}
    mha1::MHA1
    mha2::MHA2
    ln1::LN1
    ln2::LN2
    ln3::LN3
    ff::FF
    dropout::DO
    useCustomCrossAttention::Bool
end

function MATDecoderBlock(d_model::Int, nheads::Int, head_dim::Int, d_ff::Int; pdrop=0.1, useCustomCrossAttention::Bool=false)
    MATDecoderBlock(
        MultiHeadAttention(d_model => head_dim*nheads => d_model, nheads=nheads, dropout_prob=pdrop),
        MultiHeadAttention(d_model => head_dim*nheads => d_model, nheads=nheads, dropout_prob=pdrop),
        LayerNorm(d_model),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Chain(Dense(d_model, d_ff, gelu), Dense(d_ff, d_model)),
        Dropout(pdrop),
        useCustomCrossAttention,
    )
end

Flux.@layer MATDecoderBlock trainable=(mha1, mha2, ln1, ln2, ln3, ff)

function (m::MATDecoderBlock)(x, obs_rep)
    # x: (d_model, T, B)
    mask = NNlib.make_causal_mask(x, dims=2)

    if m.useCustomCrossAttention
        y, _ = m.mha1(obs_rep, x, x; mask=mask)
        # x = m.ln1(obs_rep .+ m.dropout(y))
        x = obs_rep .+ m.dropout(y)
    else
        y, _ = m.mha1(x, obs_rep, obs_rep; mask=mask)
        # x = m.ln1(x .+ m.dropout(y))
        x = x .+ m.dropout(y)
    end

    y, _ = m.mha2(x, x, x; mask=mask)
    # x = m.ln2(x .+ m.dropout(y))
    x = x .+ m.dropout(y)

    z = m.ff(x)
    # x = m.ln3(x .+ m.dropout(z))
    x = x .+ m.dropout(z)

    return x
end

struct BlockStack{T}
    blocks::T
end

Flux.@layer BlockStack trainable=(blocks)

function BlockStack(N::Integer, d_model::Int, nheads::Int, head_dim::Int, d_ff::Int;
                    pdrop=0.1, useCustomCrossAttention::Bool=false)
    blocks = ntuple(_ -> MATDecoderBlock(d_model, nheads, head_dim, d_ff;
                                         pdrop=pdrop,
                                         useCustomCrossAttention=useCustomCrossAttention),
                    Int(N))
    return BlockStack(blocks)
end

function (s::BlockStack)(x, obs_rep)
    for b in s.blocks
        x = b(x, obs_rep)
    end
    return x
end



Base.@kwdef struct MATEncoder{
    E, PE, LN, DO, BL,
    EV, PEV, LNV, DOV, BLV,
    H
}
    embedding::E
    position_encoding::PE
    ln::LN
    dropout::DO
    blocks::BL

    embedding_v::EV = nothing
    position_encoding_v::PEV = nothing
    ln_v::LNV = nothing
    dropout_v::DOV = nothing
    blocks_v::BLV = nothing

    head::H

    jointPPO::Bool = false
    useSeparateValueChain::Bool = false
end

Flux.@layer MATEncoder trainable=(embedding, position_encoding, ln, dropout, blocks, embedding_v, position_encoding_v, ln_v, dropout_v, blocks_v, head)

function (m::MATEncoder)(x)

    # zero-copy: (dm, N) -> (dm, N, 1), run, then back
    @inline function run_blocks(blocks, x::AbstractMatrix)
        x3 = reshape(x, size(x,1), size(x,2), 1)          # (dm, N, 1)
        y3 = blocks(x3)                                   # (dm, N, 1)
        return reshape(y3, size(y3,1), size(y3,2))         # (dm, N)
    end

    # already batched: do nothing
    @inline run_blocks(blocks, x::AbstractArray{T,3}) where {T} = blocks(x)

    if m.useSeparateValueChain
        vv = m.embedding_v(x)
    end
    
    x = m.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ m.position_encoding(1:N) # (dm, N, B)

    # x = m.ln(x)

    # if !(iszero(x))
    #     x = m.ln(x)
    # end

    x = m.dropout(x)                # (dm, N, B)

    rep = run_blocks(m.blocks, x)     # (dm, N, B)

    if m.useSeparateValueChain
        if m.jointPPO
            vv = vv .+ m.position_encoding_v(1:N)

            vv = m.ln_v(vv)

            # if !(iszero(vv))
            #     vv = m.ln_v(vv)
            # end

            vv = m.dropout_v(vv)                # (dm, N, B)
            vv = run_blocks(m.blocks_v, vv)     # (dm, N, B)

            sr = size(vv)
            v = m.head( reshape(vv, sr[1]*sr[2], sr[3]) ) 
            v = reshape(v, 1, 1, sr[3])                   # (1, 1, B)
            v = repeat(v, 1,sr[2],1)                   # (1, N, B)
        else
            vv = vv .+ m.position_encoding_v(1:N)
         
            # vv = m.ln_v(vv)

            # if !(iszero(vv))
            #     vv = m.ln_v(vv)
            # end

            vv = m.dropout_v(vv)                # (dm, N, B)
            vv = run_blocks(m.blocks_v, vv)     # (dm, N, B)

            v = m.head(vv)       # (1, N, B)
        end
    else
        if m.jointPPO
            sr = size(rep)
            v = m.head( reshape(rep, sr[1]*sr[2], sr[3]) ) 
            v = reshape(v, 1, 1, sr[3])                   # (1, 1, B)
            v = repeat(v, 1,sr[2],1)                   # (1, N, B)
        else
            v = m.head(rep)                   # (1, N, B)
        end
    end

    rep, v
end


Base.@kwdef struct MATDecoder{
    E, PE, LN, DO, BL, H, LOGP
}
    embedding::E
    position_encoding::PE
    ln::LN
    dropout::DO
    blocks::BL
    head::H

    logσ::LOGP

    logσ_is_network::Bool = false
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
end

Flux.@layer MATDecoder trainable=(embedding, position_encoding, ln, dropout, blocks, head, logσ)

function (m::MATDecoder)(x, obs_rep)

    # zero-copy: (dm, N) -> (dm, N, 1), run, then back
    @inline function run_blocks(blocks, x::AbstractMatrix, obs_rep::AbstractMatrix)
        x3 = reshape(x, size(x,1), size(x,2), 1)          # (dm, N, 1)
        obs_rep3 = reshape(obs_rep, size(obs_rep,1), size(obs_rep,2), 1)          # (dm, N, 1)
        y3 = blocks(x3, obs_rep3)                                   # (dm, N, 1)
        return reshape(y3, size(y3,1), size(y3,2))         # (dm, N)
    end

    # already batched: do nothing
    @inline run_blocks(blocks, x::AbstractArray{T,3}, obs_rep::AbstractArray{T,3}) where {T} = blocks(x, obs_rep)

    x = m.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ m.position_encoding(1:N) # (dm, N, B)

    # x = m.ln(x)

    # if !(iszero(x))
    #     x = m.ln(x)
    #     # x = (x.-mean(x, dims=1))./std(x, dims=1)
    # end

    x = m.dropout(x)                # (dm, N, B)

    x = run_blocks(m.blocks, x, obs_rep)     # (dm, N, B)

    x = m.head(x)                   # (1, N, B)

    # x = m.head(obs_rep[:, 1:N, :])

    if m.logσ_is_network
        raw_logσ = m.logσ(obs_rep)
    else
        if ndims(x) >= 2
            # TODO: Make it GPU friendly again (CUDA.fill or like Flux Dense Layer does it with bias - Linear Layer with freezing)
            #raw_logσ = hcat([raw_logσ for i in 1:size(μ)[2]]...)
            raw_logσ = send_to_device(device(m.logσ), Float32.(ones(size(x)))) .* m.logσ
        else
            raw_logσ = m.logσ[:]
        end
    end

    logσ = clamp.(raw_logσ, log(m.min_σ), log(m.max_σ))

    return x, logσ
end
















function create_chain_mat(;
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

function create_logσ_mat(;logσ_is_network, ns, na, use_gpu, init, nna_scale, network_depth = 2, drop_middle_layer = nothing, fun = relu, start_logσ = 0.0)

    res = nothing

    if logσ_is_network
        res = create_chain_mat(ns = ns, na = na, use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, network_depth = network_depth, drop_middle_layer = drop_middle_layer, fun = fun, tanh_end = false)
    else
        res = Matrix(Matrix(Float32.(ones(na) .* start_logσ)')')
    end

    res = use_gpu ? res |> gpu : res

    return res
end


struct CustomClipNorm <: Optimisers.AbstractRule
    omega::Float64
    p::Float64
    throw::Bool
end
CustomClipNorm(ω = 10, p = 2; throw::Bool = true) = CustomClipNorm(ω, p, throw)
  
Optimisers.init(o::CustomClipNorm, x::AbstractArray) = nothing
  
function Optimisers.apply!(o::CustomClipNorm, state, x::AbstractArray{T}, dx) where T
    nrm = Optimisers._norm(dx, o.p)

    if o.throw && !isfinite(nrm)
        throw(DomainError("gradient has $(o.p)-norm $nrm, for array $(summary(x))"))
    end
    λ = T(min(o.omega / nrm, 1))

    return state, Optimisers.@lazy dx * λ
end



struct ZeroEncoding
    hidden_size::Int
end

(embed::ZeroEncoding)(x) = zeros(Float32, embed.hidden_size, length(x))

struct SinCosPositionEmbed{A}
    pe::A  # (d_model, maxlen)
end

Flux.@layer SinCosPositionEmbed trainable=()

function SinCosPositionEmbed(d_model::Int, maxlen::Int=2048;
                             base=10000f0, offset::Int=0, T::Type=Float32)
    pe = zeros(T, d_model, maxlen)
    nfreq = cld(d_model, 2)                 # Anzahl der sin-"Kanäle"
    pos = T.(offset:offset+maxlen-1)        # 0,1,2,... (matcht Transformers.jl Beispiel)

    @inbounds for j in 0:nfreq-1
        denom = base^(T(2*j)/T(d_model))
        angle = pos ./ denom
        i_sin = 2j + 1
        pe[i_sin, :] .= sin.(angle)
        i_cos = 2j + 2
        if i_cos <= d_model
            pe[i_cos, :] .= cos.(angle)
        end
    end
    return SinCosPositionEmbed(pe)
end

# pe(N) -> (d_model, N)
(pe::SinCosPositionEmbed)(N::Int) = (N <= size(pe.pe, 2) || throw(ArgumentError("N=$N > maxlen=$(size(pe.pe,2))"));
                                    @view pe.pe[:, 1:N])

(pe::SinCosPositionEmbed)(idxs::AbstractVector{<:Integer}) = @view pe.pe[:, idxs]


function create_agent_mat(;action_space, state_space, use_gpu, rng, y, p, update_freq = 256, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, fun = leakyrelu, fun_critic = nothing, n_actors = 1, clip1 = false, n_epochs = 4, n_microbatches = 4, normalize_advantage = true, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, adaptive_weights = false, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, dim_model = 64, block_num = 1, head_num = 4, head_dim = nothing, ffn_dim = 120, drop_out = 0.1, betas = (0.99, 0.99), jointPPO = false, customCrossAttention = true, one_by_one_training = false, clip_range = 0.2f0, tanh_end = false, positional_encoding = 1, useSeparateValueChain = false, verbose = false)

    isnothing(nna_scale_critic)         &&  (nna_scale_critic = nna_scale)
    !isnothing(drop_middle_layer)        &&  (network_depth = drop_middle_layer ? 1 : 2)
    !isnothing(drop_middle_layer_critic) &&  (network_depth_critic = drop_middle_layer_critic ? 1 : 2)
    isnothing(network_depth_critic)      &&  (network_depth_critic = network_depth)
    network_depth = max(1, Int(network_depth))
    network_depth_critic = max(1, Int(network_depth_critic))
    isnothing(fun_critic)               &&  (fun_critic = fun)
    isnothing(head_dim)                 &&  (head_dim = Int(floor(dim_model/head_num)))

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
        if tanh_end
            head_decoder = Dense(dim_model, na, tanh)
        else
            head_decoder = Dense(dim_model, na)
        end
        head_decoder.weight[:] *= 0.01
        head_decoder.bias[:] *= 0.01
    else
        decoder_layers = Any[Dense(dim_model, ffn_dim, fun)]
        for _ in 3:network_depth
            push!(decoder_layers, Dense(ffn_dim, ffn_dim, fun))
        end
        if tanh_end
            push!(decoder_layers, Dense(ffn_dim, na, tanh))
        else
            push!(decoder_layers, Dense(ffn_dim, na))
        end
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
    end



    if useSeparateValueChain
        embedding_v = Dense(ns, dim_model, bias = false)
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
        embedding = Dense(ns, dim_model, bias = false),
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
        embedding = Dense(na, dim_model, bias = false),
        position_encoding = position_encoding_decoder,
        ln = LayerNorm(dim_model),
        dropout = Dropout(drop_out),
        blocks = decoder_blocks,
        head = head_decoder,
        logσ = create_logσ_mat(logσ_is_network = logσ_is_network, ns = dim_model, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale, network_depth = network_depth, fun = fun, start_logσ = start_logσ),
        logσ_is_network = logσ_is_network,
        max_σ = max_σ,
    )

    
    
    #encoder_optimizer = Optimisers.OptimiserChain(CustomClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas))
    # encoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 0.01))
    # decoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 0.01))

    # encoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.RMSProp(learning_rate))
    # decoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.RMSProp(learning_rate))

    encoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas))
    decoder_optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas))

    encoder_state_tree = Flux.setup(encoder_optimizer, encoder)
    decoder_state_tree = Flux.setup(decoder_optimizer, decoder)

    Agent(
        policy = MATPolicy(
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
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
                state = Float32 => (size(state_space)[1], n_actors),
                action = Float32 => (size(action_space)[1], n_actors),
                action_log_prob = Float32 => (n_actors),
                reward = Float32 => (n_actors),
                terminated = Bool => (n_actors,),
                truncated = Bool => (n_actors,),
                next_state = Float32 => (size(state_space)[1], n_actors),
        ),
    )
end




Base.@kwdef mutable struct MATPolicy <: AbstractPolicy
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


function prob(
    p::MATPolicy,
    state::AbstractArray,
    mask,
)
    na = size(p.decoder.embedding.weight)[2]
    batch_size = length(size(state)) == 3 ? size(state)[3] : 1

    obsrep, val = p.encoder(state)

    μ, logσ = p.decoder(zeros(Float32,na,1,batch_size), obsrep[:,1:1,:])

    for n in 2:p.n_actors
        newμ, newlogσ = p.decoder(cat(zeros(Float32,na,1,batch_size), μ, dims=2), obsrep[:,1:n,:])

        μ = cat(μ, newμ[:,end:end,:], dims=2)
        logσ = cat(logσ, newlogσ[:,end:end,:], dims=2)
    end

    StructArray{Normal}((μ, exp.(logσ)))
end


function prob(p::MATPolicy, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::MATPolicy, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::MATPolicy)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end


function (p::MATPolicy)(env::AbstractEnv)

    if p.update_step <= p.start_steps
        dist = prob(p, env)
        action = p.start_policy(env)
    else
        dist = prob(p, env)
        action = rand.(p.rng, dist)
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

    p.last_action_log_prob = log_p[:]

    action
end

function (agent::Agent{<:MATPolicy})(env::MultiThreadEnv)

    if agent.policy.update_step <= policy.start_steps
        agent.policy.start_policy(env)
    else
        dist = prob(agent.policy, env)
        action = rand.(agent.policy.rng, dist)
        if ndims(action) == 2
            action_log_prob = sum(logpdf.(dist, action), dims=1)
        else
            action_log_prob = sum(logpdf.(dist, action), dims=1)
        end
        EnrichedAction(action; action_log_prob=vec(action_log_prob))
    end
end






function update!(
    trajectory::AbstractTrajectory,
    ::MATPolicy,
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
    policy::MATPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(
        trajectory;
        state=state(env),
        action=action,
        action_log_prob=policy.last_action_log_prob
    )
end


function update!(
    trajectory::AbstractTrajectory,
    policy::MATPolicy,
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
    p::MATPolicy,
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

function update_IL(p::MATPolicy, trajectory::AbstractTrajectory)
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





function _update!(p::MATPolicy, t::Any)
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

    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))

    _, values = p.encoder(states)
    _, next_values = p.encoder(next_states)

    values = reshape(send_to_host(values), n_actors, :)
    next_values = reshape(send_to_host(next_values), n_actors, :)

    rewards = t[:reward]
    terminated = t[:terminated]
    truncated = t[:truncated]

    if p.jointPPO
        values = values[1,:]
        next_values = next_values[1,:]
        rewards = rewards[1,:]
        terminated = terminated[1,:]
        truncated = truncated[1,:]
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

    # @show advantages
    # @show select_last_dim(values, 1:n_rollout)
    # error("debug")

    actions = to_device(t[:action])
    action_log_probs = t[:action_log_prob]

    if p.jointPPO
        action_log_probs = sum(action_log_probs, dims=1)[:]
    else
        action_log_probs = reshape(action_log_probs, 1, size(action_log_probs)[1], size(action_log_probs)[2])
    end

    stop_update = false

    if p.one_by_one_training
        n_epochs = n_epochs * p.n_actors
    end

    rand_actor_inds = shuffle!(rng, Vector(1:p.n_actors))
    reverse_actor_inds = Vector(p.n_actors:-1:1)

    actor_losses = Float32[]
    critic_losses = Float32[]

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            #global s, a, r, log_p, adv

            s = to_device(collect(select_last_dim(states, inds)))
            a = to_device(collect(select_last_dim(actions, inds)))

            r = select_last_dim(returns, inds)
            log_p = select_last_dim(action_log_probs, inds)
            adv = select_last_dim(advantages, inds)

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-5 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end


            g_encoder, g_decoder = Flux.gradient(p.encoder, p.decoder) do encoder, decoder
                #global obs_rep, v′, temp_act, μ, logσ, log_p′ₐ, ratio

                obs_rep, v′ = encoder(s)
                
                # obs_rep, v′_no = p.encoder(s)
                # obs_rep_no, v′ = encoder(s)

                # parallel act
                # temp_act = cat(zeros(Float32,1,1,size(a)[3]),a[:,1:end-1,:],dims=2)
                # μ, logσ = decoder(temp_act, obs_rep)

                # auto regressive act
                μ, logσ = decoder(zeros(Float32,size(a,1),1,microbatch_size), obs_rep[:,1:1,:])

                for n in 2:p.n_actors
                    newμ, newlogσ = decoder(cat(zeros(Float32,size(a,1),1,microbatch_size), μ, dims=2), obs_rep[:,1:n,:])

                    μ = cat(μ, newμ[:,end:end,:], dims=2)
                    logσ = cat(logσ, newlogσ[:,end:end,:], dims=2)
                end
                
                log_p′ₐ = sum(normlogpdf(μ, exp.(logσ), a), dims=1)

                
                if p.jointPPO
                    log_p′ₐ = sum(log_p′ₐ, dims=2)[:]
                end

                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl && (i > 1 || epoch > 1) # only in second batch
                        if p.verbose
                            println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        end
                        stop_update = true
                    end
                end

                #adv = reshape(adv, 1, size(adv)[1], size(adv)[2])
                #r = reshape(r, 1, size(r)[1], size(r)[2])


                

                if p.jointPPO
                    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

                    surr1 = ratio .* adv
                    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))

                    critic_loss = mean((r .- v′[1,1,:]) .^ 2)

                elseif p.one_by_one_training
                    temp_index = 1 + epoch%p.n_actors
                    #actor_index = rand_actor_inds[temp_index]
                    #actor_index = reverse_actor_inds[temp_index]
                    actor_index = temp_index

                    entropy_loss = mean(size(logσ[:,actor_index,:], 1) * (log(2.0f0π) + 1) .+ sum(logσ[:,actor_index,:]; dims=1)) / 2

                    surr1 = ratio[1,actor_index,:] .* adv[actor_index,:]
                    surr2 = clamp.(ratio[1,actor_index,:], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv[actor_index,:]

                    actor_loss = -mean(min.(surr1, surr2))

                    critic_loss = mean((r[actor_index,:] .- v′[1,actor_index,:]) .^ 2)

                else
                    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

                    surr1 = ratio[1,:,:] .* adv
                    surr2 = clamp.(ratio[1,:,:], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                    actor_loss = -mean(min.(surr1, surr2))

                    critic_loss = mean((r .- v′[1,:,:]) .^ 2)
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
        actor_factor = clamp(0.5/mean_actor_loss, 0.9, 1.1)
        critic_factor = clamp(0.3/mean_critic_loss, 0.9, 1.1)
        if p.verbose
            println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
            println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
        end
        p.actor_loss_weight = w₁ * actor_factor
        p.critic_loss_weight = w₂ * critic_factor
    end

    if p.verbose
        println("---")
    end
end


function test_update(apprentice::MATPolicy, batch, μ_expert)
    
    na = size(apprentice.decoder.embedding.weight)[2]
    
    g_decoder = Flux.gradient(apprentice.decoder) do p_decoder

        obsrep, val = apprentice.encoder(batch)

        temp_act = cat(zeros(Float32,1,1,1),μ_expert[:,1:end-1,:],dims=2)

        μ, logσ = p_decoder(temp_act, obsrep[:,:,:]) # Zeros do not work here


        diff = μ - μ_expert
        mse = mean(diff.^2)

        if apprentice.verbose
            Zygote.@ignore println(mse)
        end

        return mse
    end

    Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder[1])

end
