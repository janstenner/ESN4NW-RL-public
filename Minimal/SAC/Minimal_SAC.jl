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


scriptname = "Minimal_SAC"


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end


include("./../Minimal_env.jl")


seed = Int(floor(rand()*100000))
# seed = 800

gpu_env = false



# agent tuning parameters
nna_scale = 5.0
nna_scale_critic = 2.5
network_depth = 3
fun = gelu
logσ_is_network = false
tanh_end = false
use_gpu = false
actionspace = Space(fill(-1..1, (action_dim)))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.9997f0
gamma = y
a = 3f-4 #0.2f0
t = 0.005f0
target_entropy = -1.0
use_popart = false


learning_rate = 2e-4
trajectory_length = 1_000_000
batch_size = 256
update_after = 200_000
update_freq = 50
update_loops = 3
clip_grad = 1.0
start_logσ = -1.5
automatic_entropy_tuning = true

reward_shaping = false

wind_only = false

verbose = false




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


        
        

        global agent = create_agent_sac(
                action_space = actionspace,
                state_space = env.state_space,
                rng = rng,
                y = y,
                a = a,
                t = t,
                use_gpu = use_gpu,
                update_after = update_after,
                update_freq = update_freq,
                update_loops = update_loops,
                trajectory_length = trajectory_length,
                batch_size = batch_size,
                learning_rate = learning_rate,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                network_depth = network_depth,
                fun = fun,
                logσ_is_network = logσ_is_network,
                clip_grad = clip_grad,
                start_logσ = start_logσ,
                tanh_end = tanh_end,
                automatic_entropy_tuning = automatic_entropy_tuning,
                target_entropy = target_entropy,
                use_popart = use_popart,
                verbose = verbose,)


    global hook = GeneralHook(min_best_episode = min_best_episode,
                            collect_NNA = false,
                            generate_random_init = generate_random_init,
                            collect_history = false,
                            collect_rewards_all_timesteps = false,
                            early_success_possible = true)
end



initialize_setup()

trajectories_file = joinpath(dirname(@__DIR__), "optimal_trajectories.jld2")
trajectories = isfile(trajectories_file) ? FileIO.load(trajectories_file, "trajectories") : Dict{String, Any}()
rs_key = reward_shaping ? "with_RS" : "no_RS"
optimal_trajectory = get(get(trajectories, "SAC", Dict{String, Any}()), rs_key, nothing)





function render_run(; plot_optimal = false, steps = 6000, show_training_episode = false, show_σ = false, exploration = false, return_plot = false, plot_values = true, plot_critic2 = false, critic2_diagnostics = false, json = false, new_day = true,)
    render_run_shared(;
        profile = :sac,
        plot_optimal = plot_optimal,
        steps = steps,
        show_training_episode = show_training_episode,
        show_σ = show_σ,
        exploration = exploration,
        return_plot = return_plot,
        plot_values = plot_values,
        plot_critic2 = plot_critic2,
        critic2_diagnostics = critic2_diagnostics,
        json = json,
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
