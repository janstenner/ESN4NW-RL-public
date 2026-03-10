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
using JuMP
using Ipopt
using Optimisers
#using Blink
using JSON
using UnicodePlots


scriptname = "Minimal2_PPO"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end


include("./../Minimal2_env.jl")


seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false



# agent tuning parameters
memory_size = 0
nna_scale = 5.0
nna_scale_critic = 2.5
network_depth = 3
fun = gelu
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.9999997f0
p = 0.995f0
gamma = y

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 8_000


learning_rate = 1e-4
learning_rate_critic = 2e-4
n_epochs = 5
n_microbatches = 100
logσ_is_network = false
max_σ = 1.0f0
entropy_loss_weight = 0#.1agen
clip_grad = 1.0
target_kl = Inf#0.01
clip1 = false
start_logσ = -0.6
tanh_end = false
clip_range = 0.2f0
clip_range_vf = 0.2f0

betas = (0.9, 0.99)
noise = nothing#"perlin"


normalize_advantage = true

reward_shaping = false

wind_only = false





function initialize_setup(;use_random_init = false)

    global env = GeneralEnv(do_step = do_step, 
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = actionspace,
                max_value = 1.0,
                check_max_value = "nothing",
                terminated_at_timeout = terminated_at_timeout,
                )


        
        dim = 15

        logσ = Chain(
            Dense(state_dim, dim, relu, bias = false),
            Dense(dim, dim, relu, bias = false),
            Dense(dim, 1, identity, bias = false)
        )

        logσ.layers[1].weight[:] .*= 0.2
        logσ.layers[2].weight[:] .*= 0.2
        logσ.layers[2].weight[:] = -(abs.(logσ.layers[2].weight[:]))

        approximator = ActorCritic(
            actor = GaussianNetwork(
                μ = Chain(
                    Dense(state_dim, 20, fun),
                    Dense(20, 12, fun),
                    Dense(12, 6, fun),
                    Dense(6, 1)
                ),
                logσ = [-0.5],
                logσ_is_network = false,
                max_σ = max_σ,
            ),
            critic = Chain(
                Dense(state_dim, 20, fun),
                Dense(20, 12, fun),
                Dense(12, 6, fun),
                Dense(6, 1)
            ),
            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
            optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(learning_rate, betas)),
        )

        global agent = create_agent_ppo(
                # approximator = approximator,
                action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
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
                normalize_advantage = normalize_advantage,)


    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end


initialize_setup()

trajectories_file = joinpath(dirname(@__DIR__), "optimal_trajectories.jld2")
trajectories = FileIO.load(trajectories_file, "trajectories")
rs_key = reward_shaping ? "with_RS" : "no_RS"
optimal_trajectory = trajectories["PPO"][rs_key]





function render_run(; plot_optimal = false, steps = 6000, show_training_episode = false, show_σ = false, exploration = false, return_plot = false, gae = false, plot_values = true, new_day = true)
    render_run_shared(;
        profile = :ppo,
        plot_optimal = plot_optimal,
        steps = steps,
        show_training_episode = show_training_episode,
        show_σ = show_σ,
        exploration = exploration,
        return_plot = return_plot,
        gae = gae,
        plot_values = plot_values,
        new_day = new_day,
    )
end
