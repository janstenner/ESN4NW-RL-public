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


scriptname = "Minimal"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    #println(io, "saves/*")
    println(io, "training.mp4")
end


include("./../Minimal_env.jl")


seed = Int(floor(rand()*100000))
#seed = 23038

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
p = 0.9f0
gamma = y

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 8000

critic_frozen_update_freq = 4
actor_update_freq = 1


learning_rate = 1e-4
learning_rate_critic = 2e-4
n_epochs = 5
n_microbatches = 100
logσ_is_network = false
max_σ = 1.0f0
entropy_loss_weight = 0.0f0
clip_grad = 1.0
target_kl = Inf
clip1 = false
start_logσ = -0.6
tanh_end = false
clip_range = 0.2f0
clip_range_vf = 0.2f0

λ_targets = 0.995f0
n_targets = 100

betas = (0.9, 0.99)
noise = nothing #"perlin"
noise_scale = 20
normalize_advantage = true
fear_scale = 0.4
new_loss = false#true
adaptive_weights = true
critic2_takes_action = true
use_popart = false
use_exploration_module = false
use_whole_delta_targets = true
use_critic3 = false

trajectory_size = 100_000
off_policy_update_freq = 50
off_policy_batch_size = 256

critic_frozen_factor = 0.0f0
antithetic_mean_samples = 8
zero_mean_tether_factor = 0.0f0
actorbatch_size = nothing

verbose = false

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


        
        dim = 250

        logσ = Chain(
            Dense(state_dim, dim, relu, bias = false),
            Dense(dim, dim, relu, bias = false),
            Dense(dim, 1, identity, bias = false)
        )

        logσ.layers[1].weight[:] .*= 0.2
        logσ.layers[2].weight[:] .*= 0.2
        logσ.layers[2].weight[:] = -(abs.(logσ.layers[2].weight[:]))

        critic = Chain(
                Dense(state_dim, dim, fun),
                Dense(dim, dim, fun),
                #Dense(dim, dim, fun),
                Dense(dim, 1)
            )

        critic2 = Chain(
                Dense(state_dim + 1, dim, fun),
                Dense(dim, dim, fun),
                #Dense(dim, dim, fun),
                Dense(dim, 1)
            )

        approximator = ActorCritic3(
            actor = GaussianNetwork(
                μ = Chain(
                    Dense(state_dim, dim, fun),
                    Dense(dim, dim, fun),
                    #Dense(dim, dim, fun),
                    Dense(dim, 1)
                ),
                logσ = logσ,
                logσ_is_network = true,
                max_σ = max_σ,
            ),
            critic = critic,
            critic_frozen = deepcopy(critic),
            critic2 = critic2,
            critic2_frozen = deepcopy(critic2),
            optimizer_actor = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
            optimizer_sigma = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate, betas, 1e-4, 1e-8;)),
            optimizer_critic = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
            optimizer_critic2 = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.AdamW(learning_rate_critic, betas, 1e-4, 1e-8;)),
        )

        global agent = create_agent_ppo3(
                #approximator = approximator,
                action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                update_freq = update_freq,
                critic_frozen_update_freq = critic_frozen_update_freq,
                actor_update_freq = actor_update_freq,
                learning_rate = learning_rate,
                learning_rate_critic = learning_rate_critic,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                network_depth = network_depth,
                fun = fun,
                clip1 = clip1,
                n_epochs = n_epochs,
                n_microbatches = n_microbatches,
                actorbatch_size = actorbatch_size,
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
                noise_scale = noise_scale,
                normalize_advantage = normalize_advantage,
                fear_scale = fear_scale,
                new_loss = new_loss,
                adaptive_weights = adaptive_weights,
                critic2_takes_action = critic2_takes_action,
                use_popart = use_popart,
                critic_frozen_factor = critic_frozen_factor,
                λ_targets = λ_targets,
                n_targets = n_targets,
                use_critic3 = use_critic3,
                use_exploration_module = use_exploration_module,
                use_whole_delta_targets = use_whole_delta_targets,
                antithetic_mean_samples = antithetic_mean_samples,
                zero_mean_tether_factor = zero_mean_tether_factor,
                verbose = verbose,
                trajectory_size = trajectory_size,
                off_policy_update_freq = off_policy_update_freq,
                off_policy_batch_size = off_policy_batch_size,
                )


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
optimal_trajectory = trajectories["PPO3"][rs_key]





function render_run(; plot_optimal = false, steps = 6000, show_training_episode = false, show_σ = false, exploration = false, return_plot = false, gae = true, plot_values = true, plot_critic2 = false, critic2_diagnostics = false, new_day = true,)
    render_run_shared(;
        profile = :ppo3,
        plot_optimal = plot_optimal,
        steps = steps,
        show_training_episode = show_training_episode,
        show_σ = show_σ,
        exploration = exploration,
        return_plot = return_plot,
        gae = gae,
        plot_values = plot_values,
        plot_critic2 = plot_critic2,
        critic2_diagnostics = critic2_diagnostics,
        new_day = new_day,
    )
end


function get_state(x, y)
    st = Float32[0.8; 0.0; 0.0; 0.0; 0.0; 0.0; 0.4; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.2;;]

    grid_base = Float32[0.05310250000000005, 0.06105506000000005, 0.06905967000000002, 0.07711570000000001, 0.08522229999999997]
    wind_base = Float32[0.05310250000000005, 0.06105506000000005, 0.06905967000000002, 0.07711570000000001, 0.08522229999999997]

    st[2:6] .= grid_base .+ x * 0.9
    st[8:12] .= wind_base .+ y * 0.9

    st[13] = max(0.0, st[8] - curtailment_threshold)

    st
end

function plot_critic(; return_plot = false)
    xx = collect(0:0.025:1)
    yy = xx
    
    critic_values = zeros(Float32, length(xx), length(yy))

    for (i, _) in enumerate(xx), (j, _) in enumerate(yy)
        st = get_state(i, j)

        critic_value = agent.policy.approximator.critic(st)[1]

        critic_values[i, j] = critic_value
    end
    

    p = plot(surface(x=xx, y=yy, z=critic_values), Layout(
        scene = attr(
            xaxis_title="grid price",
            yaxis_title="wind power",
            zaxis_title="Critic Value"
        )
    ))

    if return_plot
        return p
    else
        display(p)
    end
end


function plot_trajectory()
    t = agent.trajectory
    AC = agent.policy.approximator
    states = collect(flatten_batch(t[:state]))
    actions = collect(flatten_batch(t[:action]))

    values = AC.critic(states)
    next_values = values + AC.critic2(vcat(states, actions))

    advantages, returns = generalized_advantage_estimation(
        t[:reward],
        values,
        next_values,
        y,
        p;
        dims=2,
        terminal=t[:terminal]
    )

    colorscale = [[0, "rgb(255, 0, 0)"], [0.5, "rgb(255, 255, 255)"], [1, "rgb(0, 255, 0)"], ]

    layout = Layout(
                    plot_bgcolor = "white",
                    font=attr(
                        family="Arial",
                        size=16,
                        color="black"
                    ),
                    showlegend = true,
                    legend=attr(x=0.5, y=-0.1, orientation="h", xanchor="center"),
                    xaxis = attr(gridcolor = "#E0E0E0FF",
                                linecolor = "#888888"),
                    yaxis = attr(gridcolor = "#E0E0E0FF",
                                linecolor = "#888888",
                                range=[0,1]),
                    yaxis2 = attr(
                        overlaying="y",
                        side="right",
                        titlefont_color="orange",
                        #range=[-1, 1]
                    ),
                )

    to_plot = AbstractTrace[]
    

    push!(to_plot, scatter(y=advantages[:], name="Advantage", yaxis = "y2",
            mode="lines+markers",
            marker=attr(
                color=advantages[:],               # array of numbers
                cmin = -0.01,
                cmid = 0.0,
                cmax = 0.01,
                colorscale=colorscale,
                showscale=false
            ),
            line=attr(color = "rgba(200, 200, 200, 0.3)")))
    
    push!(to_plot, scatter(y=t[:reward][:], name="Reward", yaxis = "y2"))
    


    push!(to_plot, scatter(y=values[:], name="Critic Values", yaxis = "y2"))
    push!(to_plot, scatter(y=next_values[:], name="Next Value", yaxis = "y2"))
    push!(to_plot, scatter(y=returns[:], name="Return", yaxis = "y2"))


    push!(to_plot, scatter(y=states[1,:], name="Load Left"))
    push!(to_plot, scatter(y=states[2,:], name="Grid Price"))

    push!(to_plot, scatter(y=t[:action][:], name="WindCORE utilization 1"))

    push!(to_plot, scatter(y=states[6,:], name="Wind Power 1"))

    push!(to_plot, scatter(y=Float32.(t[:terminal][:]), name="Terminal"))

    plott = plot(Vector(to_plot), layout)

    display(plott)
end

include("./same_day_trainer.jl")
include("./critic_problem.jl")
