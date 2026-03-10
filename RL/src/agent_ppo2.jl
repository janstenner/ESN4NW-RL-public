Base.@kwdef mutable struct ModulationModule
    # Configuration parameters
    p_explore::Float32 = 0.07   # probability to enter explore mode each time step
    min_amp::Float32    = 0.2    # minimum exploration amplitude
    max_amp::Float32    = 1.6    # maximum exploration amplitude
    min_width::Int      = 1      # minimum explore duration (in steps)
    max_width::Int      = 13     # maximum explore duration (in steps)

    # Internal state
    state::Symbol       = :idle  # current mode (:idle or :explore)
    last::Float32       = 0.0    # last output value

    # Explore-mode parameters (set upon entering explore)
    amp::Float32        = 1.0    # current amplitude
    width::Int          = 0      # total duration of current explore burst
    sigma::Float32      = 1.0    # Gaussian standard deviation
    center::Float32     = 0.0    # Gaussian center
    t::Int              = 0      # time step within current explore burst
    min_value::Float32 = 0.0    # minimum value for the output
end


function (mm::ModulationModule)(use_exploration_module = true)
    
    if use_exploration_module
        if mm.state == :idle
            # In idle mode, output zero
            mm.last = mm.min_value
            # Possibly switch to explore
            if rand() < mm.p_explore
                mm.state  = :explore
                mm.width  = rand(mm.min_width:mm.max_width)
                mm.amp    = mm.min_amp + rand() * (mm.max_amp - mm.min_amp)
                mm.sigma  = mm.width / 6
                mm.center = mm.width / 2
                mm.t      = 0
            end

        elseif mm.state == :explore
            # Compute Gaussian-shaped output
            mm.last = mm.min_value + mm.amp * exp(-((mm.t - mm.center)^2) / (2 * mm.sigma^2))
            mm.t += 1
            # Return to idle after the burst completes
            if mm.t > mm.width
                mm.state = :idle
            end
        end
    else
        mm.last = 1.0f0
    end

    return mm.last
end


mutable struct PopArt
    dense      # die letzte Dense‑Schicht (beliebiger Typ, z. B. Flux.Dense)
    μ          # laufender Mittelwert der Targets (beliebiger Float‑Typ)
    σ          # laufende Standardabweichung (beliebiger Float‑Typ)
    β          # Glättungsfaktor für die Statistik (beliebiger Float‑Typ)
end


function PopArt(in_dim; β=Float32(1e-3), init=glorot_uniform)
    dense_layer = Dense(in_dim, 1; init=init)
    return PopArt(dense_layer, 0.0f0, 1.0f0, β)
end

# ────────────────────────────────────────────────────────────────────────────
# Functor‑Definition: damit Flux.params und gradient() wissen, welche Felder
# sie traversieren sollen. μ und σ werden hier nicht als trainierbar markiert!
# ────────────────────────────────────────────────────────────────────────────
Flux.@layer PopArt

# Wir überschreiben, welche Felder wirklich trainiert werden:
Flux.trainable(p::PopArt) = (; dense = p.dense)


function (p::PopArt)(x)
    ŷ̃ = p.dense(x)          # interne "normierte" Vorhersage
    return p.σ .* ŷ̃ .+ p.μ  # unnormierter Output
end

# ────────────────────────────────────────────────────────────────────────────
# Statistik‑Update + Preserve‑Rescale
# Aufruf: update!(popart_layer, target_vector)
# ────────────────────────────────────────────────────────────────────────────
function update!(p::PopArt, targets)
    # Alte Werte sichern
    μ_old, σ_old = p.μ, p.σ

    # 1) Batch‑Statistiken
    batch_mean = mean(targets)
    batch_var  = mean((targets .- batch_mean).^2)

    # 2) Exponentiell gleitende Updates
    p.μ += p.β * (batch_mean - p.μ)
    p.σ = sqrt((1 - p.β) * (p.σ^2) + p.β * batch_var)

    # 3) Preserve‑Rescale der Dense‑Gewichte
    #    Damit bleibt σ_old * ŷ̃ + μ_old == p.σ * ŷ̃ + p.μ
    scale = σ_old / p.σ
    p.dense.weight .*= scale
    p.dense.bias .= (σ_old .* p.dense.bias .+ μ_old .- p.μ) ./ p.σ

    return nothing
end


Base.@kwdef mutable struct ActorCritic2{A,C,C2}
    actor::A
    critic::C # value function approximator
    critic_frozen::C
    critic2::C2 # action value function approximator minus r
    critic2_frozen::C2
    optimizer_actor = ADAM()
    optimizer_sigma = ADAM()
    optimizer_critic = ADAM()
    optimizer_critic2 = ADAM()
    actor_state_tree = nothing
    sigma_state_tree = nothing
    critic_state_tree = nothing
    critic2_state_tree = nothing
end

@forward ActorCritic2.critic device

function create_critic_PPO2(;ns, na, use_gpu, init, nna_scale, network_depth = 2, drop_middle_layer = nothing, fun = relu, is_critic2 = false, popart = false)
    if !isnothing(drop_middle_layer)
        network_depth = drop_middle_layer ? 1 : 2
    end
    network_depth = max(1, Int(network_depth))

    nna_size_critic = Int(floor(20 * nna_scale))

    if is_critic2
        input_size = ns + na
    else
        input_size = ns
    end
    
    if popart
        last = PopArt(nna_size_critic; init = init)
    else
        last = Dense(nna_size_critic, 1; init = init)
    end

    layers = Any[Dense(input_size, nna_size_critic, fun; init = init)]
    for _ in 2:network_depth
        push!(layers, Dense(nna_size_critic, nna_size_critic, fun; init = init))
    end
    push!(layers, last)
    n = Chain(layers...)


    model = use_gpu ? n |> gpu : n

    model
end


function create_agent_ppo2(;action_space, state_space, use_gpu, rng, y, p, update_freq = 2000, approximator = nothing, nna_scale = 1, nna_scale_critic = nothing, network_depth = 2, network_depth_critic = nothing, drop_middle_layer = nothing, drop_middle_layer_critic = nothing, learning_rate = 0.00001, learning_rate_critic = nothing, fun = relu, fun_critic = nothing, tanh_end = false, n_envs = 1, clip1 = false, n_epochs = 10, n_microbatches = 1, actorbatch_size=100, normalize_advantage = false, logσ_is_network = false, start_steps = -1, start_policy = nothing, max_σ = 2.0f0, actor_loss_weight = 1.0f0, critic_loss_weight = 0.5f0, entropy_loss_weight = 0.00f0, critic_regularization_loss_weight=0.01f0, logσ_regularization_loss_weight=0.01f0, adaptive_weights = false, clip_grad = 0.5, target_kl = 100.0, start_logσ = 0.0, betas = (0.9, 0.999), clip_range = 0.2f0, clip_range_vf = 0.2f0, noise = nothing, noise_scale = 90, fear_factor = 1.0, fear_scale = 0.5, new_loss = true, use_exploration_module = false, verbose = false, )

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

    critic = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic)
    critic_frozen = deepcopy(critic)

    critic2 = create_critic_PPO2(ns = ns, na = na, use_gpu = use_gpu, init = init, nna_scale = nna_scale_critic, network_depth = network_depth_critic, fun = fun_critic, is_critic2 = true)
    critic2_frozen = deepcopy(critic2)

    Agent(
        policy = PPOPolicy2(
            approximator = isnothing(approximator) ? ActorCritic2(
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
                optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
                optimizer_sigma = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas)),
                optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas)),
                optimizer_critic2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas)),
            ) : approximator,
            γ = y,
            λ = p,
            clip_range = clip_range,
            clip_range_vf = clip_range_vf,
            max_grad_norm = 0.5f0,
            n_epochs = n_epochs,
            n_microbatches = n_microbatches,
            actorbatch_size = actorbatch_size,
            actor_loss_weight = actor_loss_weight,
            critic_loss_weight = critic_loss_weight,
            entropy_loss_weight = entropy_loss_weight,
            critic_regularization_loss_weight = critic_regularization_loss_weight,
            logσ_regularization_loss_weight = logσ_regularization_loss_weight,
            adaptive_weights = adaptive_weights,
            dist = Normal,
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
            fear_factor = fear_factor,
            fear_scale = fear_scale,
            new_loss = new_loss,
            mm = mm,
            use_exploration_module = use_exploration_module,
            verbose = verbose,
        ),
        trajectory = 
        CircularArrayTrajectory(;
                capacity = update_freq,
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



"""
    PPOPolicy2(;kwargs)

# Keyword arguments

- `approximator`,
- `γ = 0.99f0`,
- `λ = 0.95f0`,
- `clip_range = 0.2f0`,
- `max_grad_norm = 0.5f0`,
- `n_microbatches = 4`,
- `n_epochs = 4`,
- `actor_loss_weight = 1.0f0`,
- `critic_loss_weight = 0.5f0`,
- `entropy_loss_weight = 0.01f0`,
- `dist = Categorical`,
- `rng = Random.GLOBAL_RNG`,

If `dist` is set to `Categorical`, it means it will only work
on environments of discrete actions. To work with environments of continuous
actions `dist` should be set to `Normal` and the `actor` in the `approximator`
should be a `GaussianNetwork`. Using it with a `GaussianNetwork` supports 
multi-dimensional action spaces, though it only supports it under the assumption
that the dimensions are independent since the `GaussianNetwork` outputs a single
`μ` and `σ` for each dimension which is used to simplify the calculations.
"""


mutable struct PPOPolicy2{A<:ActorCritic2,D,R} <: AbstractPolicy
    approximator::A
    γ::Float32
    λ::Float32
    clip_range::Float32
    clip_range_vf::Union{Nothing,Float32}
    max_grad_norm::Float32
    n_microbatches::Int
    actorbatch_size::Int
    n_epochs::Int
    actor_loss_weight::Float32
    critic_loss_weight::Float32
    entropy_loss_weight::Float32
    critic_regularization_loss_weight::Float32
    logσ_regularization_loss_weight::Float32
    adaptive_weights::Bool
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
    fear_factor
    fear_scale
    new_loss::Bool
    use_exploration_module::Bool
    mm
    critic_target
    verbose::Bool
    last_action_log_prob::Vector{Float32}
    last_sigma::Vector{Float32}
    last_mu::Vector{Float32}
end

function PPOPolicy2(;
    approximator,
    update_freq,
    n_random_start=0,
    update_step=0,
    γ=0.99f0,
    λ=0.95f0,
    clip_range=0.2f0,
    clip_range_vf = 0.2f0,
    max_grad_norm=0.5f0,
    n_microbatches=1,
    actorbatch_size = 32,
    n_epochs=10,
    actor_loss_weight=1.0f0,
    critic_loss_weight=0.5f0,
    entropy_loss_weight=0.01f0,
    critic_regularization_loss_weight=0.01f0,
    logσ_regularization_loss_weight=0.01f0,
    adaptive_weights = true,
    dist=Normal,
    rng=Random.GLOBAL_RNG,
    clip1 = false,
    normalize_advantage = false,
    start_steps = -1,
    start_policy = nothing,
    target_kl = 100.0,
    noise = nothing,
    noise_sampler = nothing,
    noise_scale = 90.0,
    noise_step = 0,
    fear_factor = 1.0f0,
    fear_scale = 0.5f0,
    new_loss = true,
    mm = ModulationModule(),
    use_exploration_module = false,
    critic_target = 0.0f0,
    verbose = false,
)
    PPOPolicy2{typeof(approximator),dist,typeof(rng)}(
        approximator,
        γ,
        λ,
        clip_range,
        clip_range_vf,
        max_grad_norm,
        n_microbatches,
        actorbatch_size,
        n_epochs,
        actor_loss_weight,
        critic_loss_weight,
        entropy_loss_weight,
        critic_regularization_loss_weight,
        logσ_regularization_loss_weight,
        adaptive_weights,
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
        fear_factor,
        fear_scale,
        new_loss,
        use_exploration_module,
        mm,
        critic_target,
        verbose,
        [0.0],
        [0.0],
        [0.0],
    )
end

function prob(
    p::PPOPolicy2{<:ActorCritic2{<:GaussianNetwork},Normal},
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

function prob(p::PPOPolicy2{<:ActorCritic2,Categorical}, state::AbstractArray, mask)
    logits = p.approximator.actor(send_to_device(device(p.approximator), state))
    if !isnothing(mask)
        logits .+= ifelse.(mask, 0.0f0, typemin(Float32))
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

function prob(p::PPOPolicy2, env::MultiThreadEnv)
    mask = nothing
    prob(p, state(env), mask)
end

function prob(p::PPOPolicy2, env::AbstractEnv)
    s = state(env)
    # s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    mask = nothing
    prob(p, s, mask)
end

function (p::PPOPolicy2)(env::MultiThreadEnv)
    result = rand.(p.rng, prob(p, env))
    if p.clip1
        clamp!(result, -1.0, 1.0)
    end
    result
end



function (p::PPOPolicy2)(env::AbstractEnv)

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        dist = prob(p, env)

        modulation_value = p.mm()
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
    policy::PPOPolicy2,
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
    policy::PPOPolicy2,
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
    p::PPOPolicy2,
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

function update_IL(p::PPOPolicy2, trajectory::AbstractTrajectory)
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









"""
    sample_negatives_far(x_batch; m=size(x_batch,2), δ=0.1,
                         max_attempts=1_000, sample_fn=()->randn(eltype(x_batch), size(x_batch,1)))

Draw up to `m` vectors in ℝⁿ (same n as `x_batch`) so that each new vector is at least
Euclidean distance `δ` from *every* column of `x_batch`.  By default samples
from N(0,1). Throws an error if it fails `max_attempts` times for any sample.
"""
function sample_negatives_far(x_batch::AbstractMatrix{T};
                              m::Int=size(x_batch,2),
                              δ::Real=0.1,
                              max_attempts::Int=1_000,
                              sample_fn::Function=()->randn(eltype(x_batch), size(x_batch,1))) where {T}

    δ2 = δ^2
    x_neg = Vector{Vector{T}}() 

    for i in 1:m
        attempt = 0
        while max_attempts > attempt
            attempt += 1
            cand = sample_fn()
            # compute squared distances to every column in x_batch
            d2 = sum((x_batch .- cand).^2; dims=1)
            if minimum(d2) ≥ δ2
                push!(x_neg, cand)
                break
            end
        end
    end

    return hcat(x_neg...)
end


function prepare_values(values, terminated)
    offset_value = 0.0

    for i in length(values):-1:1
        if terminated[i]
            offset_value = values[i]
        end
        values[i] -= offset_value
    end

    return values
end





# --- QUANTIL BERECHNUNG FÜR FEAR FACTOR --- #



# --- optionaler Helfer für gewichtete Quantile innerhalb eines Batches ---
function weighted_quantile(x::AbstractVector, w::AbstractVector, p::Real)
    @assert length(x) == length(w)
    idx = sortperm(x)
    xs = x[idx]
    ws = w[idx] .+ eps()             # Nullgewichte vermeiden
    ws ./= sum(ws)
    cdf = cumsum(ws)
    k = searchsortedfirst(cdf, p)
    return xs[clamp(k, 1, length(xs))]
end


"""
    robust_quantiles(epsilons, adv_mags; p=0.9, weighted=true)

Gibt ein NamedTuple mit:
- q_eps  : p-Quantil von epsilons
- q_adv  : p-Quantil von |adv_mags|
- wq_eps : (optional) A-gew. p-Quantil von epsilons

Hinweis: `epsilons[i]` und `adv_mags[i]` sollten zeitlich zusammenpassen
(z.B. pro Epoche gemessene Kennzahlen).
"""
function robust_quantiles(epsilons::AbstractVector, adv_mags::AbstractVector; p=0.9, weighted::Bool=true)
    @assert length(epsilons) == length(adv_mags) "Längen passen nicht zusammen"

    # Filtere Nicht-Finite
    mask = map(isfinite, epsilons) .& map(isfinite, adv_mags)
    E = collect(epsilons[mask])
    A = abs.(collect(adv_mags[mask]))

    if isempty(E)
        return (q_eps = NaN, q_adv = NaN, wq_eps = NaN)
    end

    # Ungewichtete Quantile
    q_eps = quantile(E, p)
    q_adv = quantile(A, p)

    if !weighted
        return (q_eps = q_eps, q_adv = q_adv)
    end

    # A-gewichtetes p-Quantil von E (einfacher, robuster Algorithmus)
    # 1) Sortiere nach E
    idx = sortperm(E)
    Es = E[idx]
    Ws = A[idx] .+ eps() # mini-offset gegen Nullgewichte
    Ws ./= sum(Ws)

    # 2) Finde kleinste Stelle, wo die kumulative Gewichtssumme ≥ p ist
    cdf = cumsum(Ws)
    k = searchsortedfirst(cdf, p)
    k = clamp(k, 1, length(Es))
    wq_eps = Es[k]

    return (q_eps = q_eps, q_adv = q_adv, wq_eps = wq_eps)
end

# --- Collector für Batch-Quantile während einer Epoche ---
mutable struct BatchQuantileCollector
    eps_qs::Vector{Float64}   # pro Batch: p-Quantil von |r-1|
    adv_qs::Vector{Float64}   # pro Batch: p-Quantil von |A|
end
BatchQuantileCollector() = BatchQuantileCollector(Float64[], Float64[])

"""
    update!(c, ratio, adv; p=0.9, within_batch_weighted=false)

Ermittelt pro Batch das p-Quantil von |ratio-1| (optional A-gewichtet)
und das p-Quantil von |adv|. Schreibt die Werte in den Collector.
"""
function update!(c::BatchQuantileCollector, ratio, adv; p=0.9, within_batch_weighted::Bool=false)
    eps = abs.(ratio .- 1)
    A   = abs.(adv)

    q_eps_batch = within_batch_weighted ? weighted_quantile(eps, A, p) : quantile(eps, p)
    q_adv_batch = quantile(A, p)

    push!(c.eps_qs, float(q_eps_batch))
    push!(c.adv_qs, float(q_adv_batch))
    return nothing
end

"""
    finalize(c; p_over_epochs=0.9, weighted=true)

Aggregiert die über die Epoche gesammelten Batch-Quantile weiter zu
robusten Kennzahlen (Quantil über die Batch-Quantile; optional A-gewichtet).
"""
function finalize(c::BatchQuantileCollector; p_over_epochs=0.9, weighted::Bool=true)
    return robust_quantiles(c.eps_qs, c.adv_qs; p=p_over_epochs, weighted=weighted)
end



function _update!(p::PPOPolicy2, t::Any)
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
    w₄ = p.critic_regularization_loss_weight
    w₅ = p.logσ_regularization_loss_weight
    D = device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_rollout = size(t[:terminated])
    
    microbatch_size = Int(floor(n_envs * n_rollout ÷ n_microbatches))
    actorbatch_size = p.actorbatch_size

    n = length(t)
    states = to_device(t[:state])
    next_states = to_device(t[:next_state])
    actions = to_device(t[:action])

    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))
    next_states_flatten_on_host = flatten_batch(select_last_dim(t[:next_state], 1:n))

    values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)

    #values = prepare_values(values, t[:terminated])

    mus = AC.actor.μ(states_flatten_on_host)
    offsets = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), mus) )) , n_envs, :)

    # advantages = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), flatten_batch(actions)) )) , n_envs, :) - offsets


    q_values = reshape(send_to_host( AC.critic2( vcat(flatten_batch(states), flatten_batch(actions)) )) , n_envs, :)

    deltas = q_values - offsets

    advantages, returns = generalized_advantage_estimation(
        deltas,
        zeros(Float32, size(deltas)),
        zeros(Float32, size(deltas)),
        γ,
        λ;
        dims=2,
        terminated=t[:terminated],
        truncated=t[:truncated],
    )

    # returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    advantages = to_device(advantages)

    # if p.normalize_advantage
    #     advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
    # end

    positive_advantage_indices = findall(>(0), vec(advantages))


    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)
    explore_mod = to_device(t[:explore_mod])

    stop_update = false

    actor_losses = Float32[]
    critic_losses = Float32[]
    critic2_losses = Float32[]
    entropy_losses = Float32[]





    if isnothing(AC.actor_state_tree) || isnothing(AC.sigma_state_tree) || isnothing(AC.critic_state_tree) || isnothing(AC.critic2_state_tree)
        if p.verbose
            println("________________________________________________________________________")
            println("Reset Optimizers")
            println("________________________________________________________________________")
        end
        AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor.μ)
        AC.sigma_state_tree = Flux.setup(AC.optimizer_sigma, AC.actor.logσ)
        AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
        AC.critic2_state_tree = Flux.setup(AC.optimizer_critic2, AC.critic2)
    end



    next_states = to_device(flatten_batch(t[:next_state]))
    rewards = to_device(t[:reward])
    terminated = Matrix{Bool}(to_device(t[:terminated]))
    truncated = Matrix{Bool}(to_device(t[:truncated]))
    done = Matrix{Bool}(terminated .| truncated)

    v_ref = AC.critic_frozen( to_device(flatten_batch(t[:state])) )[:] 

    q_ref = AC.critic2_frozen( vcat(to_device(flatten_batch(t[:state])), to_device(flatten_batch(t[:action]))) )[:] 


    next_values = reshape( AC.critic( next_states ), n_envs, :)
    targets = lambda_truncated_targets(rewards, done, next_values, γ)[:]

    next_q_values = reshape( AC.critic2( vcat(next_states, AC.actor.μ( next_states) ) ), n_envs, :)
    targets_q = td_lambda_targets(rewards, terminated, truncated, next_q_values, γ; λ=0.7f0)[:]

    collector = BatchQuantileCollector()
    

    rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))

    for epoch in 1:n_epochs

        
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            inds_actor = inds[1:clamp(actorbatch_size, 1, length(inds))]

            #inds = positive_advantage_indices

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            ns = to_device(collect(select_last_dim(next_states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))
            exp_m = to_device(collect(select_last_dim(explore_mod, inds)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            #r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds]
            adv = vec(advantages)[inds]

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

                μ, logσ = actor(s)

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

                
                #fear = (abs.((ratio .- 1)) + ones(size(ratio))).^1.3 .* p.fear_factor
                fear = (ratio .- 1).^2 .* p.fear_factor

                # if !(isempty(s_neg))
                #     v_neg = critic(s_neg)
                #     critic_regularization = mean((v_neg .- p.critic_target) .^ 2)
                # else
                #     critic_regularization = 0.0f0
                # end

                # if AC.actor.logσ_is_network
                #     logσ_neg = AC.actor.logσ(s_neg)
                #     logσ_regularization  = mean((logσ_neg .- log(AC.actor.max_σ)) .^ 2)	
                # else
                #     logσ_regularization  = 0.0f0
                # end

                if p.new_loss
                    actor_loss_values = ((ratio .* adv) - fear)  .* exp_m[:]
                    actor_loss = -mean(actor_loss_values[inds_actor])
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
                critic_loss = bellman + 0.4 * fr_term # .* exp_m[:])
                # critic2_loss = mean(((nv .- nv′) .^ 2)) # .* exp_m[:])


                loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss #+ w₄ * critic_regularization #+ w₅ * logσ_regularization


                ignore() do
                    # println("---------------------")
                    # println(critic_loss)
                    # println(critic_regularization)
                    # println("---------------------")
                    push!(actor_losses, w₁ * actor_loss)
                    push!(critic_losses, w₂ * critic_loss)
                    #push!(critic2_losses, w₂ * critic2_loss)
                    push!(entropy_losses, -w₃ * entropy_loss)
                    # push!(critic_regularization_losses, w₄ * critic_regularization)
                    # push!(logσ_regularization_losses, w₅ * logσ_regularization)


                    #println(maximum(abs.(ratio .- 1)))

                    update!(collector, ratio, adv; p=0.9, within_batch_weighted=true)

                    # polyak update critic target
                    # p.critic_target = p.critic_target * 0.9 + (maximum(r) - 0.1) * 0.1
                end

                loss
            end
            
            if !stop_update
                if (p.update_step / p.update_freq) % 4 == 0
                    Flux.update!(AC.actor_state_tree, AC.actor.μ, g_actor.μ)
                    Flux.update!(AC.sigma_state_tree, AC.actor.logσ, g_actor.logσ)
                end
                Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
            else
                break
            end

            #update!(AC.critic.layers[end], tar) 




            nv′ = vec(next_values)[inds]
            rew = vec(rewards)[inds]
            ter = vec(terminated)[inds]

            tar = rew + γ * nv′ .* (1 .- ter)
            tar = vec(targets_q)[inds]

            old_v2 = vec(q_values)[inds]

            g_critic2 = Flux.gradient(AC.critic2) do critic2
                critic2_values = critic2(vcat(s,a)) |> vec

                if isnothing(clip_range_vf) || clip_range_vf == 0.0
                    values_pred2 = critic2_values
                else
                    # clipped value function loss, from OpenAI SpinningUp implementation
                    Δ = critic2_values .- old_v2
                    values_pred2 = old_v2 .+ clamp.(Δ, -clip_range_vf, clip_range_vf)
                end

                
                bellman = mean(((tar .- values_pred2) .^ 2))
                fr_term = mean((values_pred2 .- q_ref[inds]) .^ 2)
                critic2_loss = bellman + 0.4 * fr_term # .* exp_m[:]

                ignore() do
                    push!(critic2_losses, w₂ * critic2_loss)
                end

                critic2_loss
            end

            Flux.update!(AC.critic2_state_tree, AC.critic2, g_critic2[1])

            #update!(AC.critic2.layers[end], tar) 

        end
    end


    #println(p.update_step / p.update_freq)

    if (p.update_step / p.update_freq) % 4 == 0
        if p.verbose
            println("CRITIC FROZEN UPDATE")
        end
        AC.critic_frozen = deepcopy(AC.critic)
        AC.critic2_frozen = deepcopy(AC.critic2)
    end


    # everything here is just magnitude (abs), not real mean

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
    end
    # println("mean logσ regularization loss: $(mean_logσ_regularization_loss)")
    # println("mean critic regularization loss: $(mean_critic_regularization_loss)")

    q = finalize(collector; p_over_epochs=0.9, weighted=true)

    # q.q_eps   : 0.9-Quantil der (pro Batch) 0.9-Quantile von |r-1|
    # q.q_adv   : 0.9-Quantil der (pro Batch) 0.9-Quantile von |A|
    # q.wq_eps  : A-gewichtetes 0.9-Quantil über die Batch-Quantile von |r-1|

    
    if p.verbose
        println("0.9-Quantil excitement: $(q.q_adv)")
        println("weighted 0.9-Quantil |r-1|: $(q.wq_eps)")
    end

    if p.adaptive_weights && (p.update_step / p.update_freq) % 4 == 0

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
