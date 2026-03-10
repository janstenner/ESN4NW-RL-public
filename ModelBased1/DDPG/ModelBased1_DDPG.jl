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


scriptname = "ModelBased1_DDPG"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
end


include("./../ModelBased1_env.jl")


seed = Int(floor(rand()*100000))
# seed = 578

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
y = 0.9997f0
gamma = y
p = 0.995f0
batch_size = 256
start_steps = -1
start_policy = ZeroPolicy(actionspace)
update_after = 200_000
update_freq = 50
update_loops = 3
reset_stage = POST_EPISODE_STAGE
learning_rate = 5e-6
learning_rate_critic = 3e-6
clip_grad = 0.4
act_limit = 1.0
act_noise = 0.05
trajectory_length = 1_000_000

reward_shaping = false










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


    global agent = create_agent(mono = true,
                        action_space = actionspace,
                        state_space = env.state_space,
                        use_gpu = use_gpu, 
                        rng = rng,
                        y = y, p = p, batch_size = batch_size, 
                        start_steps = start_steps, 
                        start_policy = start_policy,
                        update_after = update_after, 
                        update_freq = update_freq,
                        update_loops = update_loops,
                        reset_stage = reset_stage,
                        act_limit = act_limit, 
                        act_noise = act_noise,
                        nna_scale = nna_scale,
                        nna_scale_critic = nna_scale_critic,
                        network_depth = network_depth,
                        fun = fun,
                        memory_size = memory_size,
                        trajectory_length = trajectory_length,
                        learning_rate = learning_rate,
                        learning_rate_critic = learning_rate_critic,
                        clip_grad = clip_grad,)

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
optimal_trajectory = trajectories["DDPG"][rs_key]




function render_run(use_best = false; exploration = true, new_day = true)
    render_run_shared(; profile = :ddpg, use_best = use_best, exploration = exploration, new_day = new_day)
end
