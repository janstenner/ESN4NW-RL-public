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

scriptname = "Continuous1_MAT_Categorial"

dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end

# Must be set before including the environment.
dist = Categorical
multi_agent = true

include("./../Continuous1_env.jl")

seed = Int(floor(rand() * 100000))
# seed = 800

gpu_env = false

# agent tuning parameters
memory_size = 0
nna_scale = 5.0
nna_scale_critic = 2.5
network_depth = 3
network_depth_critic = 3
fun = gelu
use_gpu = false

# MAT-specific parameters
dim_model = 30
block_num = 1
head_num = 2
head_dim = 15
ffn_dim = 30
drop_out = 0.0
positional_encoding = 1
customCrossAttention = true
jointPPO = false
one_by_one_training = false
useSeparateValueChain = true

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.95f0
gamma = y

start_steps = -1
start_policy = env -> rand(rng, 1:n_discrete_actions, 1, n_jobs)

update_freq = 8_000

learning_rate = 1e-4
n_epochs = 5
n_microbatches = 50
logσ_is_network = false
max_σ = 1.0f0
entropy_loss_weight = 0.01f0
clip_grad = 1.0
target_kl = Inf
clip1 = false
start_logσ = -0.6
clip_range = 0.2f0

betas = (0.9, 0.99)

normalize_advantage = true

function initialize_setup(; use_random_init = false)
    global env = create_env()

    global agent = create_agent_mat_categorial(
        action_space = actionspace,
        state_space = env.state_space,
        use_gpu = use_gpu,
        rng = rng,
        y = y,
        p = p,
        update_freq = update_freq,
        learning_rate = learning_rate,
        nna_scale = nna_scale,
        nna_scale_critic = nna_scale_critic,
        network_depth = network_depth,
        network_depth_critic = network_depth_critic,
        fun = fun,
        n_actors = n_jobs,
        clip1 = clip1,
        n_epochs = n_epochs,
        n_microbatches = n_microbatches,
        logσ_is_network = logσ_is_network,
        max_σ = max_σ,
        entropy_loss_weight = entropy_loss_weight,
        clip_grad = clip_grad,
        target_kl = target_kl,
        start_logσ = start_logσ,
        clip_range = clip_range,
        betas = betas,
        normalize_advantage = normalize_advantage,
        start_steps = start_steps,
        start_policy = start_policy,
        dim_model = dim_model,
        block_num = block_num,
        head_num = head_num,
        head_dim = head_dim,
        ffn_dim = ffn_dim,
        drop_out = drop_out,
        positional_encoding = positional_encoding,
        customCrossAttention = customCrossAttention,
        jointPPO = jointPPO,
        one_by_one_training = one_by_one_training,
        useSeparateValueChain = useSeparateValueChain,
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
