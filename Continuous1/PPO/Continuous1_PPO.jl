using LinearAlgebra
using IntervalSets
using StableRNGs
using SparseArrays
using FFTW
using PlotlyJS
using FileIO, JLD2
using Flux
using Random
using RL
using DataFrames
using Statistics
using Optimisers
using UnicodePlots
using Distributions: Categorical

scriptname = "Continuous1_PPO"

# dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end

# Set policy distribution before including the environment config.
dist = Categorical

include("./../Continuous1_env.jl")

seed = Int(floor(rand() * 100000))
# seed = 800

gpu_env = false

# agent tuning parameters
memory_size = 0
nna_scale = 5.0
nna_scale_critic = 2.5
network_depth = 3
fun = gelu
use_gpu = false

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.95f0
gamma = y

start_steps = -1
start_policy = dist == Categorical ? (env -> rand(rng, 1:n_discrete_actions)) : ZeroPolicy(actionspace)

update_freq = 8_000

learning_rate = 1e-4
learning_rate_critic = 2e-4
n_epochs = 5
n_microbatches = 50
logσ_is_network = false
max_σ = 1.0f0
entropy_loss_weight = 0.0
clip_grad = 1.0
target_kl = Inf
clip1 = false
start_logσ = -0.6
tanh_end = true
clip_range = 0.2f0
clip_range_vf = 0.2f0

betas = (0.9, 0.99)
noise = nothing

normalize_advantage = true

function initialize_setup(; use_random_init = false)
    global env = create_env()

    global agent = create_agent_ppo(
        action_space = actionspace,
        state_space = env.state_space,
        use_gpu = use_gpu,
        rng = rng,
        y = y,
        p = p,
        update_freq = update_freq,
        learning_rate = learning_rate,
        learning_rate_critic = learning_rate_critic,
        nna_scale = nna_scale,
        nna_scale_critic = nna_scale_critic,
        network_depth = network_depth,
        fun = fun,
        clip1 = clip1,
        n_epochs = n_epochs,
        n_microbatches = n_microbatches,
        dist = dist,
        logσ_is_network = logσ_is_network,
        max_σ = max_σ,
        entropy_loss_weight = entropy_loss_weight,
        clip_grad = clip_grad,
        target_kl = target_kl,
        start_logσ = start_logσ,
        tanh_end = tanh_end,
        clip_range = clip_range,
        clip_range_vf = clip_range_vf,
        betas = betas,
        noise = noise,
        normalize_advantage = normalize_advantage,
    )

    global hook = GeneralHook(
        min_best_episode = min_best_episode,
        collect_NNA = false,
        generate_random_init = generate_random_init,
        collect_history = false,
        collect_rewards_all_timesteps = false,
        early_success_possible = true,
    )
end

initialize_setup()
