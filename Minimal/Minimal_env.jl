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
using MathOptInterface



n_windCORES = 1
n_turbines = 1


validation_scores = []


te = 1440.0
dt = 5.0
t0 = 0.0
min_best_episode = 1
terminated_at_timeout = true


# default - will be overwritten in most training scripts
reward_shaping = false
reward_shaping_beta = 1.0f0



# action vector dim - contains the percentage of maximum power the HPC in the turbine will use for the duration of next time step
action_dim = n_windCORES


# Curtailment threshold
curtailment_threshold = 0.4

# state vector

# - amount of computation left (starts at 1.0 and goes to 0.0)
# - price of energy from the grid (last 5 steps)
# - current curtailment threshold
# - wind stituation at every turbine (last 5 steps) plus current wind power minus curtailment threshold
# - current time

history_steps = 5

function generate_wind(; same_day = false)
    global history_steps, te, dt

    wind_steps = Int(te/dt) + history_steps

    if same_day
        rand1 = 0.6
        rand2 = 0.2
        rand3 = 0.3
        rand4 = 0.3
        rand5 = 0.3
        rand6 = 0.3
        rand7 = 0.3
        rand8 = 0.1
    else
        rand1 = rand()
        rand2 = randn()
        rand3 = rand()
        rand4 = rand()
        rand5 = randn()
        rand6 = rand()
        rand7 = rand()
        rand8 = randn()
    end

    #@show rand1, rand2, rand3, rand4, rand5, rand6, rand7, rand8

    wind_constant_day = rand1
    deviation = 1/5

    result = sign(rand2) * sin.(collect(LinRange(rand3*3+1, 4+rand4*4, wind_steps)))

    for i in 1:4
        result += sign(rand5) * sin.(collect(LinRange(rand6+4, 5+rand7*i*4, wind_steps)))
    end

    result .-= minimum(result)
    result ./= maximum(result)
    result .*= deviation

    day_wind = sign(rand8) * sin.(collect(LinRange(wind_constant_day*2*pi, 2+wind_constant_day*2*pi, wind_steps)))
    day_wind .+= 1.0
    day_wind ./= 4
    day_wind .+= 0.25


    result .+= day_wind

    clamp!(result, -1.0, 1.0)

    result
end

function generate_grid_price(; same_day = false)
    global history_steps, te, dt

    grid_price_steps = Int(te/dt) + history_steps

    if same_day
        rand1 = 0.15
        rand2 = 0.25
        rand3 = 0.35
        rand4 = 0.20
        rand5 = 0.10
        rand6 = 0.30
    else
        rand1 = rand()
        rand2 = rand()
        rand3 = rand()
        rand4 = rand()
        rand5 = rand()
        rand6 = rand()
    end

    # Base day curve: highest at beginning/end, lowest around midday
    t = collect(LinRange(0, 2 * pi, grid_price_steps))
    day_curve = 0.25 .+ 0.5 .* (sin.(t .+ pi / 2) .+ 1) ./ 2

    # Mild variations (less random than wind)
    a1 = 0.06 + 0.02 * rand4
    a2 = 0.04 + 0.02 * rand5
    a3 = 0.02 + 0.01 * rand6
    w1 = 1.0 + 0.5 * rand1
    w2 = 2.0 + 0.5 * rand2
    w3 = 3.0 + 0.5 * rand3

    variation = a1 .* sin.(w1 .* t .+ 2 * pi * rand2) .+
                a2 .* sin.(w2 .* t .+ 2 * pi * rand3) .+
                a3 .* sin.(w3 .* t .+ 2 * pi * rand1)

    result = day_curve .+ variation

    clamp!(result, 0.0, 1.0)

    result
end

"""
Generate one wind signal and one grid-price signal and plot both over daytime (0-24h).
Returns `(wind_day, grid_price_day, plott)`.
"""
function plot_generated_day_signals(; same_day = false, return_plot = false)
    global te, dt

    wind_signal = generate_wind(; same_day = same_day)
    grid_price_signal = generate_grid_price(; same_day = same_day)

    day_steps = Int(round(te / dt))
    day_steps <= length(wind_signal) || error("Wind signal shorter than one day.")
    day_steps <= length(grid_price_signal) || error("Grid price signal shorter than one day.")

    # Keep only the actual day horizon (without the prepended history part).
    wind_day = wind_signal[end - day_steps + 1:end]
    grid_price_day = grid_price_signal[end - day_steps + 1:end]
    x_hours = collect(0:dt / 60:(te - dt) / 60)

    traces = AbstractTrace[
        scatter(x = x_hours, y = wind_day, name = "Wind Signal"),
        scatter(x = x_hours, y = grid_price_day, name = "Grid Price Signal"),
    ]

    layout = Layout(
        title = "Generated Wind and Grid Price Signals",
        xaxis = attr(title = "Time of Day [h]"),
        yaxis = attr(title = "Normalized Signal Value"),
        #plot_bgcolor = "white",
    )

    plott = plot(traces, layout)
    if return_plot
        return wind_day, grid_price_day, plott
    end

    display(plott);
end

include_history_steps = 1
include_gradients = 2

function create_state(; env = nothing, compute_left = 1.0, step = 0, generate_day = true, same_day = false)
    global wind, grid_price, curtailment_threshold, history_steps, dt, include_history_steps, include_gradients


    if isnothing(env)
        y = [1.0]

        if generate_day
            wind = [generate_wind(; same_day = same_day) for i in 1:n_turbines]
            grid_price = generate_grid_price(; same_day = same_day)
        end

        time = 0.0

    else
        y = [compute_left]

        step = env.steps + 1

        time = (env.time + dt) / env.te

    end


    #test
    # y = []
    # for i in 1:n_turbines
    #     for j in history_steps:-1:(1 + (history_steps - include_history_steps))
    #         push!(y, wind[i][j+step])
    #     end
    # end
    # push!(y, time)
    # return Float32.(y)


    for i in history_steps:-1:(1 + (history_steps - include_history_steps))
        push!(y, grid_price[i+step])
    end

    if include_gradients > 0
        g1 = 500 * (grid_price[history_steps+step] - grid_price[history_steps+step-1])/dt
        push!(y, g1)
        if include_gradients > 1

            # if (grid_price[history_steps+step] == 1.0 && grid_price[history_steps+step-2] != 1.0) || (grid_price[history_steps+step] != 1.0 && grid_price[history_steps+step-2] == 1.0)
            #     g2 = 0.0
            # else
            #     g2 = 50_000 * (grid_price[history_steps+step] - 2*grid_price[history_steps+step-1] + grid_price[history_steps+step-2])/(dt^2)
            # end
            g2 = 50_000 * (grid_price[history_steps+step] - 2*grid_price[history_steps+step-1] + grid_price[history_steps+step-2])/(dt^2)


            push!(y, g2)
        end
    end


    push!(y, curtailment_threshold)

    for i in 1:n_turbines

        for j in history_steps:-1:(1 + (history_steps - include_history_steps))
            push!(y, wind[i][j+step])
        end

        if include_gradients > 0
            g1 = 500 * (wind[i][history_steps+step] - wind[i][history_steps+step-1])/dt
            push!(y, g1)
            if include_gradients > 1
                g2 = 50_000 * (wind[i][history_steps+step] - 2*wind[i][history_steps+step-1] + wind[i][history_steps+step-2])/(dt^2)
                push!(y, g2)
            end
        end

        push!(y, max(0.0, wind[i][history_steps+step] - curtailment_threshold))
    end

    push!(y, time)


    Float32.(y)
end

y0 = create_state()
state_dim = length(y0)

sim_space = Space(fill(0..1, (state_dim)))

include("./Validation_Minimal.jl")



function smoothedReLu(x)
    x *= 100_000

    if x <= 0.0
        result =  0.0
    elseif x <= 0.5
        result =   x^2
    else
        result =   x - 0.25
    end

    return result / 100_000
end


function softplus_shifted(x)
    factor = 700
    log( 1 + exp(factor * (x - 0.006)) ) / factor
end

# xx = collect(-1:0.001:1)
# plot(scatter(y=softplus_shifted.(xx), x=xx))

reward_scale_factor = 100

function calculate_day(action, env, step = nothing; reward_shaping = reward_shaping)
    global curtailment_threshold, wind, grid_price, history_steps

    if !isnothing(env)
        global wind_only

        compute_left = env.y[1]
        step = env.steps
    else
        compute_left = nothing
        wind_only = false
    end

    step += history_steps

    compute_power = 0.0
    for i in 1:n_windCORES
        compute_power += action[i]*0.01/n_windCORES
    end

    compute_left_before = compute_left

    if !isnothing(env)
        # subtracting the computed load
        compute_power_used = min(compute_left, compute_power)
        compute_left -= compute_power
        compute_left = max(compute_left, 0.0)

        if compute_left == 0.0
            env.terminated = true
        end
    else
        compute_power_used = compute_power
    end

    compute_left_after = compute_left

    #normalizing
    compute_power_used *= 100/n_turbines

    # reward calculation
    if wind_only
        tempreward = 0.0
        power_for_free = 0.0

        for i in 1:n_turbines
            # curtailment energy onlny when wind is above 0.4
            temp_free_power = (wind[i][step-1] - curtailment_threshold)
            temp_free_power = max(0.0, temp_free_power)

            power_for_free += temp_free_power
        end

        tempreward += abs( sum(action) - power_for_free )
        reward = - tempreward/288.0
    else

        power_for_free = 0.0

        for i in 1:n_turbines

            # curtailment energy onlny when wind is above 0.4
            temp_free_power = (wind[i][step-1] - curtailment_threshold)
            temp_free_power = max(0.0, temp_free_power)

            power_for_free += temp_free_power
        end

        #special_reward = max(0.1 - abs(power_for_free - compute_power_used), 0)
        special_reward = 0

        compute_power_used -= power_for_free
        #compute_power_used = max(0.0, compute_power_used)
        compute_power_used = softplus_shifted(compute_power_used)

        #normalizing
        compute_power_used *= (n_turbines * 0.01)
        
        reward1 = compute_power_used * grid_price[step-1]
        reward = - reward1

        if !isnothing(env) 
            if (env.time + env.dt) >= env.te 
                reward -= compute_left * 1.0
                compute_left_after = 0.0f0 
            end
        end
    end

    if reward_shaping
        # potential based reward shaping
        beta = reward_shaping_beta
        reward += beta * (compute_left_before - compute_left_after - (gamma-1) * compute_left_after)

        reward *= reward_scale_factor
    end

    return reward, compute_left_after
end


function do_step(env; reward_shaping = false)
    global wind_only
    
    reward, compute_left = calculate_day(env.p, env; reward_shaping = reward_shaping)

    #env.reward = [ -(reward^2)]
    env.reward = [reward]
    
    y = create_state(; env = env, compute_left = compute_left, step = env.steps + 1)

    return y
end

function reward_function(env)
    return env.reward
end



function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = y0
    else
        y = env.y
    end

    return reshape(y, length(y), 1)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing) 
    if isnothing(env)
        action =  action0
    else
        action = env.action
    end

    action = (action .+1) .*0.5

    clamp!(action, 0.0, 1.0)

    return action
end



function generate_random_init(; same_day = false)
    
    y0 = create_state(; same_day = same_day)

    env.y0 = deepcopy(y0)
    env.y = deepcopy(y0)
    env.state = env.featurize(; env = env)

    y0
end


episode_done(env) = is_terminated(env) || is_truncated(env)

function _render_run_state_is_newday(profile::Symbol)
    profile in (:ppo3, :flowppo, :sac2, :ppo)
end

function _render_run_critic2_takes_action(profile::Symbol)
    if profile in (:ppo3, :flowppo) && isdefined(@__MODULE__, :critic2_takes_action)
        return critic2_takes_action
    end
    return true
end

function _render_run_heatmap_values(profile::Symbol, states, actions; mus = nothing, normalize = false)
    n_states = length(states)
    n_actions = length(actions)
    values = zeros(Float32, n_actions, n_states)
    subtract_mu = profile in (:ppo2, :sac, :sac2)

    for (i, st) in enumerate(states)
        state_mat = repeat(st, 1, n_actions)
        action_mat = reshape(actions, 1, :)

        if profile in (:sac, :sac2)
            q1 = agent.policy.qnetwork1(vcat(state_mat, action_mat))[:]
            q2 = agent.policy.qnetwork2(vcat(state_mat, action_mat))[:]
            q_vals = min.(q1, q2)

            if subtract_mu
                mu = isnothing(mus) ? agent.policy.actor.μ(st) : mus[i]
                mu = ndims(mu) == 2 ? mu[:, 1] : mu
                mu_q = min(
                    agent.policy.qnetwork1(vcat(st, mu))[1],
                    agent.policy.qnetwork2(vcat(st, mu))[1]
                )
                values[:, i] = q_vals .- mu_q
            else
                values[:, i] = q_vals
            end
        else
            critic2 = agent.policy.approximator.critic2
            takes_action = _render_run_critic2_takes_action(profile)
            inputs = takes_action ? vcat(state_mat, action_mat) : state_mat
            c2_vals = critic2(inputs)[:]

            if subtract_mu && takes_action
                mu = isnothing(mus) ? agent.policy.approximator.actor.μ(st) : mus[i]
                mu = ndims(mu) == 2 ? mu[:, 1] : mu
                mu_val = critic2(vcat(st, mu))[1]
                values[:, i] = c2_vals .- mu_val
            else
                values[:, i] = c2_vals
            end
        end
    end

    if normalize
        values = (values .- mean(values)) ./ clamp(std(values), 1e-8, 1000.0)
    end

    values
end

function render_run_shared(;
    profile::Symbol,
    use_best = false,
    plot_optimal = false,
    steps = 6000,
    show_training_episode = false,
    show_σ = false,
    exploration = false,
    return_plot = false,
    gae = true,
    plot_values = true,
    plot_critic2 = false,
    critic2_diagnostics = false,
    json = false,
    new_day = true,
)
    if show_training_episode
        training_episode = length(hook.rewards)
    end

    ddpg_mode = profile == :ddpg
    ppo_mode = profile in (:ppo, :ppo2, :ppo3, :flowppo)
    sac_mode = profile in (:sac, :sac2)
    ppo3_like = profile in (:ppo3, :flowppo)
    sac2_mode = profile == :sac2

    start_steps_backup = nothing
    update_after_backup = nothing
    update_step_backup = nothing

    if ddpg_mode
        if use_best
            copyto!(agent.policy.behavior_actor, hook.bestNNA)
        end
        if hasproperty(agent.policy, :start_steps)
            start_steps_backup = agent.policy.start_steps
            agent.policy.start_steps = -1
        end
        if hasproperty(agent.policy, :update_after)
            update_after_backup = agent.policy.update_after
            agent.policy.update_after = 100000
        end
        if hasproperty(agent.policy, :update_step)
            update_step_backup = agent.policy.update_step
            agent.policy.update_step = 0
        end
    end

    global rewards = Float64[]
    reward_sum = 0.0
    xx = collect(dt/60:dt/60:te/60)

    global results_run = Dict("rewards" => Float64[], "loadleft" => Float64[])
    for k in 1:max(n_windCORES, n_turbines)
        results_run["hpc$k"] = Float64[]
        if k <= n_windCORES
            results_run["σ$k"] = Float64[]
        end
    end

    values = Float32[]
    next_values = Float32[]
    values2 = Float32[]
    values3 = Float32[]
    offset_values = Float32[]
    q_values = Float32[]
    q1 = Float32[]
    q2 = Float32[]
    states = Vector{Any}()
    mus = Vector{Any}()
    sigmas = Vector{Any}()
    terminated_flags = Bool[]
    truncated_flags = Bool[]
    done_flags = Bool[]

    global run_logs = []
    global currentDF = DataFrame()

    if _render_run_state_is_newday(profile) && !new_day
        reset!(env)

        if ppo3_like
            y0 = create_state(; generate_day = false)
            env.y0 = deepcopy(y0)
            env.y = deepcopy(y0)
            env.state = env.featurize(; env = env)

            global day_trajectory = CircularArrayTrajectory(;
                capacity = 288,
                state = Float32 => (size(env.state_space)[1], 1),
                action = Float32 => (size(env.action_space)[1], 1),
                action_log_prob = Float32 => (1),
                reward = Float32 => (1),
                explore_mod = Float32 => (1),
                terminal = Bool => (1,),
                next_state = Float32 => (size(env.state_space)[1], 1),
            )
        elseif sac2_mode
            y0 = create_state(; generate_day = false)
            env.y0 = deepcopy(y0)
            env.y = deepcopy(y0)
            env.state = env.featurize(; env = env)

            global day_trajectory = CircularArrayTrajectory(;
                capacity = 288,
                state = Float32 => (size(env.state_space)[1], 1),
                action = Float32 => (size(env.action_space)[1], 1),
                reward = Float32 => (1),
                terminal = Bool => (1,),
            )
        else
            y0 = create_state(; generate_day = false)
            env.y0 = deepcopy(y0)
            env.y = deepcopy(y0)
            env.state = env.featurize(; env = env)
        end
    else
        reset!(env)
        generate_random_init()
    end

    if _render_run_state_is_newday(profile)
        global all_states = zeros(289, state_dim)
        all_states[1, :] = env.state
    end

    while !episode_done(env)
        action = nothing
        μ = nothing
        σ = nothing
        step_dict = Dict()

        if ddpg_mode
            action = exploration ? agent(env) : agent.policy.behavior_actor(env)
        elseif sac_mode
            action = exploration ? agent(env) : agent.policy.actor.μ(env.state)
            push!(mus, agent.policy.actor.μ(env.state)[1])
            if json
                step_dict["step"] = env.steps
                step_dict["state"] = env.state
                step_dict["action"] = action
            end
            push!(q1, agent.policy.qnetwork1(vcat(env.state, action))[1])
            push!(q2, agent.policy.qnetwork2(vcat(env.state, action))[1])
            if json
                step_dict["q1"] = q1[end]
                step_dict["q2"] = q2[end]
            end
        else
            if exploration
                action = agent(env)
                if hasproperty(agent.policy, :last_mu)
                    μ = agent.policy.last_mu[1]
                end
                if hasproperty(agent.policy, :last_sigma)
                    σ = agent.policy.last_sigma
                end
            else
                prob_temp = prob(agent.policy, env)
                action = prob_temp.μ
                μ = prob_temp.μ[1]
                σ = prob_temp.σ

                if ppo3_like
                    if ndims(action) == 2
                        log_p = vec(sum(normlogpdf(μ, σ, action), dims=1))
                    else
                        log_p = normlogpdf(μ, σ, action)
                    end
                    agent.policy.last_action_log_prob = log_p
                end
            end
        end

        if ppo_mode
            if isnothing(μ)
                if hasproperty(agent.policy, :last_mu)
                    μ = agent.policy.last_mu[1]
                else
                    μ = action[1]
                end
            end
            if isnothing(σ) && hasproperty(agent.policy, :last_sigma)
                σ = agent.policy.last_sigma
            end
            push!(mus, μ)
            push!(sigmas, σ)
            push!(states, env.state)
        end

        if profile == :ppo
            push!(values, agent.policy.approximator.critic(env.state)[1])
        elseif profile == :ppo2
            push!(offset_values, agent.policy.approximator.critic2(vcat(env.state, μ))[1])
            push!(q_values, agent.policy.approximator.critic2(vcat(env.state, action))[1])
        elseif profile == :ppo3
            push!(values, agent.policy.approximator.critic(env.state)[1])
            c2_input = _render_run_critic2_takes_action(profile) ? vcat(env.state, action) : env.state
            push!(values2, agent.policy.approximator.critic2(c2_input)[1])
            if isdefined(@__MODULE__, :use_critic3) && use_critic3
                push!(values3, agent.policy.approximator.critic3(env.state)[1])
            end
        end

        temp_state = deepcopy(env.state)
        env(action; reward_shaping = reward_shaping)

        term = is_terminated(env)
        trunc = is_truncated(env)
        done = term || trunc
        push!(terminated_flags, term)
        push!(truncated_flags, trunc)
        push!(done_flags, done)

        if _render_run_state_is_newday(profile)
            all_states[env.steps + 1, :] = env.state
        end

        if sac_mode && json
            step_dict["next_state"] = env.state
            step_dict["terminated"] = term
            step_dict["truncated"] = trunc
            step_dict["reward"] = env.reward[1]
            push!(run_logs, step_dict)
        end

        if ddpg_mode
            for k in 1:n_turbines
                push!(results_run["hpc$k"], env.p[k])
            end
        elseif profile == :ppo
            for k in 1:n_windCORES
                push!(results_run["hpc$k"], env.p[k])
                if !isnothing(σ)
                    push!(results_run["σ$k"], σ[k])
                end
            end
            push!(next_values, agent.policy.approximator.critic(env.state)[1])
        else
            for k in 1:n_windCORES
                push!(results_run["hpc$k"], clamp((action[1] + 1) * 0.5, 0, 1))
                if ppo_mode && !isnothing(σ)
                    push!(results_run["σ$k"], σ[k])
                end
            end
        end

        push!(results_run["rewards"], env.reward[1])
        push!(results_run["loadleft"], env.y[1])
        reward_sum += mean(env.reward)

        if ppo3_like && !new_day
            push!(day_trajectory;
                state = temp_state,
                action = action,
                action_log_prob = agent.policy.last_action_log_prob,
                reward = env.reward[:],
                explore_mod = 1.0f0,
                terminal = done,
                next_state = env.state,
            )
        elseif sac2_mode && !new_day
            push!(day_trajectory;
                state = temp_state,
                action = action,
                reward = env.reward[:],
                terminal = done,
            )
        end

        if ddpg_mode
            tmp = DataFrame()
            insertcols!(tmp, :timestep => env.steps)
            insertcols!(tmp, :action => [vec(env.action)])
            insertcols!(tmp, :p => [send_to_host(env.p)])
            insertcols!(tmp, :y => [send_to_host(env.y)])
            insertcols!(tmp, :reward => [reward(env)])
            if hasproperty(hook, :currentDF)
                append!(hook.currentDF, tmp)
            else
                append!(currentDF, tmp)
            end
        end
    end

    if ddpg_mode
        if use_best
            copyto!(agent.policy.behavior_actor, hook.currentNNA)
        end
        if !isnothing(start_steps_backup)
            agent.policy.start_steps = start_steps_backup
        end
        if !isnothing(update_after_backup)
            agent.policy.update_after = update_after_backup
        end
        if !isnothing(update_step_backup)
            agent.policy.update_step = update_step_backup
        end
    end

    println(reward_sum)

    colorscale = [[0, "rgb(255, 0, 0)"], [0.5, "rgb(255, 255, 255)"], [1, "rgb(0, 255, 0)"]]
    layout = Layout(
        plot_bgcolor = "white",
        font = attr(family = "Arial", size = 16, color = "black"),
        showlegend = true,
        legend = attr(x = 0.5, y = -0.1, orientation = "h", xanchor = "center"),
        xaxis = attr(gridcolor = "#E0E0E0FF", linecolor = "#888888"),
        yaxis = attr(gridcolor = "#E0E0E0FF", linecolor = "#888888"),
        yaxis2 = attr(overlaying = "y", side = "right", titlefont_color = "orange"),
    )

    if ddpg_mode
        layout.plot_bgcolor = "#f1f3f7"
        layout.yaxis = attr(range = [0, 1])
    end

    if show_training_episode
        layout.title = "Evaluation Episode after $(training_episode) Training Episodes"
    end

    x_for(y) = xx[1:length(y)]
    to_plot = AbstractTrace[]

    if ddpg_mode
        push!(to_plot, scatter(y = results_run["rewards"], name = "Reward", yaxis = "y2"))
        push!(to_plot, scatter(y = results_run["loadleft"], name = "Load Left"))
        push!(to_plot, scatter(y = grid_price, name = "Grid Price"))
        for k in 1:n_turbines
            push!(to_plot, scatter(y = results_run["hpc$k"], name = "WindCORE utilization $k"))
            push!(to_plot, scatter(y = wind[k], name = "Wind Power $k"))
        end
    else
        if show_σ
            for k in 1:n_windCORES
                push!(to_plot, scatter(x = x_for(results_run["σ$k"]), y = results_run["σ$k"], name = "σ$k", yaxis = "y2"))
            end
        elseif gae && profile in (:ppo, :ppo2, :ppo3)
            if profile == :ppo
                advantages, _ = generalized_advantage_estimation(
                    Float32.(results_run["rewards"]),
                    Float32.(values),
                    Float32.(next_values),
                    y,
                    p;
                    terminated = terminated_flags,
                    truncated = truncated_flags,
                )
                push!(to_plot, scatter(
                    x = x_for(advantages),
                    y = advantages,
                    name = "Advantage",
                    yaxis = "y2",
                    mode = "lines+markers",
                    marker = attr(color = advantages, cmin = -0.01, cmid = 0.0, cmax = 0.01, colorscale = colorscale, showscale = false),
                    line = attr(color = "rgba(200, 200, 200, 0.3)"),
                ))
            elseif profile == :ppo2
                deltas = Float32.(q_values .- offset_values)
                advantages, _ = generalized_advantage_estimation(
                    deltas,
                    zeros(Float32, length(deltas)),
                    zeros(Float32, length(deltas)),
                    y,
                    p;
                    terminated = terminated_flags,
                    truncated = truncated_flags,
                )
                if isdefined(@__MODULE__, :normalize_advantage) && normalize_advantage
                    advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
                end
                push!(to_plot, scatter(
                    x = x_for(results_run["hpc1"]),
                    y = results_run["hpc1"],
                    name = "Advantage",
                    mode = "markers",
                    marker = attr(color = advantages, cmin = -1.0, cmid = 0.0, cmax = 1.0, colorscale = colorscale, showscale = false),
                    line = attr(color = "rgba(200, 200, 200, 0.3)"),
                ))
                push!(to_plot, scatter(x = x_for(offset_values), y = offset_values, name = "Values", yaxis = "y2"))
                push!(to_plot, scatter(x = x_for(q_values), y = q_values, name = "Next Values", yaxis = "y2"))
            elseif profile == :ppo3
                if isdefined(@__MODULE__, :use_whole_delta_targets) && use_whole_delta_targets
                    local deltas
                    if isdefined(@__MODULE__, :use_critic3) && use_critic3 && !isempty(values3)
                        deltas = Float32.(values2 .- values3)
                    else
                        mean_c2 = RL.antithetic_mean(agent.policy.approximator.actor, agent.policy.approximator.critic2, reduce(hcat, states))[:]
                        deltas = Float32.(values2 .- mean_c2)
                    end
                    advantages, _ = generalized_advantage_estimation(
                        deltas,
                        zeros(Float32, length(deltas)),
                        zeros(Float32, length(deltas)),
                        y,
                        p;
                        terminated = terminated_flags,
                        truncated = truncated_flags,
                    )
                else
                    advantages, _ = generalized_advantage_estimation(
                        Float32.(results_run["rewards"]),
                        Float32.(values),
                        Float32.(values2),
                        y,
                        p;
                        terminated = terminated_flags,
                        truncated = truncated_flags,
                    )
                end
                if isdefined(@__MODULE__, :normalize_advantage) && normalize_advantage
                    advantages = (advantages .- mean(advantages)) ./ clamp(std(advantages), 1e-8, 1000.0)
                end
                push!(to_plot, scatter(
                    x = x_for(results_run["hpc1"]),
                    y = results_run["hpc1"],
                    name = "Advantage",
                    mode = "markers",
                    marker = attr(color = advantages, cmin = -1.0, cmid = 0.0, cmax = 1.0, colorscale = colorscale, showscale = false),
                    line = attr(color = "rgba(200, 200, 200, 0.3)"),
                ))

                if isdefined(@__MODULE__, :use_critic3) && use_critic3 && !isempty(values3)
                    push!(to_plot, scatter(x = x_for(values2), y = values2 .- values3, name = "Values2-Values3", yaxis = "y2"))
                    push!(to_plot, scatter(x = x_for(values2), y = values2, name = "Values2", yaxis = "y2"))
                    push!(to_plot, scatter(x = x_for(values3), y = values3, name = "Values3", yaxis = "y2"))
                else
                    mean_c2 = RL.antithetic_mean(agent.policy.approximator.actor, agent.policy.approximator.critic2, reduce(hcat, states))[:]
                    push!(to_plot, scatter(x = x_for(values2), y = values2 .- mean_c2, name = "Values2-mean_c2", yaxis = "y2"))
                    push!(to_plot, scatter(x = x_for(values2), y = values2, name = "Values2", yaxis = "y2"))
                    push!(to_plot, scatter(x = x_for(mean_c2), y = mean_c2, name = "mean_c2", yaxis = "y2"))
                end
            end
        else
            if profile == :ppo3
                push!(to_plot, scatter(x = x_for(values), y = values, name = "Values", yaxis = "y2"))
            end
            push!(to_plot, scatter(x = x_for(results_run["rewards"]), y = results_run["rewards"], name = "Reward", yaxis = "y2"))
        end

        if plot_values
            if profile in (:sac, :sac2)
                push!(to_plot, scatter(x = x_for(q1), y = min.(q1, q2), name = "min.(q1, q2)", yaxis = "y2"))
            elseif profile == :ppo
                push!(to_plot, scatter(x = x_for(values), y = values, name = "Critic Value", yaxis = "y2"))
            end
        end

        push!(to_plot, scatter(x = x_for(results_run["loadleft"]), y = results_run["loadleft"], name = "Load Left"))
        push!(to_plot, scatter(x = xx, y = grid_price[history_steps:end], name = "Grid Price"))

        for k in 1:n_windCORES
            line_attr = profile in (:ppo2, :ppo3) ? attr(color = "rgba(200, 200, 200, 0.3)") : nothing
            if isnothing(line_attr)
                push!(to_plot, scatter(x = x_for(results_run["hpc$k"]), y = results_run["hpc$k"], name = "WindCORE utilization $k"))
            else
                push!(to_plot, scatter(x = x_for(results_run["hpc$k"]), y = results_run["hpc$k"], name = "WindCORE utilization $k", line = line_attr))
            end
        end

        for k in 1:n_turbines
            push!(to_plot, scatter(x = xx, y = wind[k][history_steps:end], name = "Wind Power $k"))
        end
    end

    if plot_optimal
        global optimal_actions = optimize_day(steps)
        global optimal_rewards = evaluate(optimal_actions; collect_rewards = true)

        for k in 1:n_windCORES
            push!(to_plot, scatter(x = xx, y = (optimal_actions[k, :] .+1) .*0.5, name = "Optimal HPC$k"))
        end

        push!(to_plot, scatter(x = xx, y = optimal_rewards, name = "Optimal Reward", yaxis = "y2"))


        println("")
        println("--------------------------------------------")
        println("AGENT:   $reward_sum")
        println("IPOPT:   $(sum(optimal_rewards))")
        println("--------------------------------------------")
    end

    plott = plot(Vector(to_plot), layout)
    if return_plot
        return plott
    else
        display(plott)
    end

    if plot_critic2 && profile != :ppo
        if isempty(states)
            return
        end

        colorscale2 = [[0.0, "rgb(50, 0, 50)"], [0.25, "rgb(200, 0, 0)"], [0.5, "rgb(210, 210, 0)"], [0.75, "rgb(0, 210, 0)"], [1.0, "rgb(140, 255, 255)"]]
        layout2 = Layout(plot_bgcolor = "#f1f3f7", coloraxis = attr(cmid = 0, colorscale = colorscale2))
        actions_grid = collect(-1:0.02:1)
        states_for_diag = states
        xx_diag = collect(Float32, xx)

        if critic2_diagnostics
            new_states = Any[]
            temp_state = deepcopy(states_for_diag[1])
            temp_state[2] = 1.0f0
            wind_index = 2 + include_history_steps - 1 + include_gradients + 2

            for i in 3:wind_index-2
                temp_state[i] = 0.0f0
            end
            temp_state[wind_index] = 0.0f0
            for i in wind_index+1:length(temp_state)-2
                temp_state[i] = 0.0f0
            end
            temp_state[end-1] = clamp(temp_state[6] - curtailment_threshold, 0.0f0, 1.0f0)
            push!(new_states, deepcopy(temp_state))

            xx_diag = Float32[0.0f0]
            for _ in 1:288
                temp_state[wind_index] += 1.0f0 / 288.0f0
                temp_state[end-1] = clamp(temp_state[wind_index] - curtailment_threshold, 0.0f0, 1.0f0)
                push!(new_states, deepcopy(temp_state))
                push!(xx_diag, temp_state[wind_index])
            end
            states_for_diag = new_states
        end

        normalize_hm = profile in (:ppo2, :sac, :sac2)
        hm = _render_run_heatmap_values(profile, states_for_diag, actions_grid; mus = (isempty(mus) ? nothing : mus), normalize = normalize_hm)
        min_val = -maximum(abs.(hm))

        for (i, st) in enumerate(states_for_diag)
            if critic2_diagnostics
                idx = clamp(searchsortedfirst(actions_grid, st[end-1] * 2 - 1), 1, length(actions_grid))
                idx2 = findmax(hm[:, i])[2]
                hm[idx2, i] = -min_val
                hm[idx, i] = min_val
            else
                mu_val = isempty(mus) ? actions_grid[findmax(hm[:, i])[2]] : (mus[i] isa Number ? mus[i] : mus[i][1])
                idx = clamp(searchsortedfirst(actions_grid, mu_val), 1, length(actions_grid))
                idx2 = findmax(hm[:, i])[2]
                hm[idx2, i] = -min_val
                hm[idx, i] = min_val
            end
        end

        display(plot(PlotlyJS.heatmap(x = xx_diag[1:size(hm, 2)], y = actions_grid, z = hm, coloraxis = "coloraxis"), layout2))
    end
end



# IPOPT Score for same_day: -0.16820833350564535

function train(use_random_init = true; visuals = false, num_steps = 10_000, inner_loops = 10, optimal_trainings  = 0, outer_loops = 8000, only_wind_steps = 0, json = false, reward_shaping = reward_shaping, plot_runs = true, same_day = false)
    global wind_only, optimal_trajectory
    wind_only = false
    
    rm(dirpath * "/training_frames/", recursive=true, force=true)
    mkdir(dirpath * "/training_frames/")
    frame = 1

    if visuals
        colorscale = [[0, "rgb(34, 74, 168)"], [0.5, "rgb(224, 224, 180)"], [1, "rgb(156, 33, 11)"], ]
        ymax = 30
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmin = 0, cmid = 1, cmax = 2, colorscale = colorscale),
            )
    end

    if use_random_init
        hook.generate_random_init = generate_random_init
    else
        hook.generate_random_init = false
    end


    if optimal_trainings > 0
        @assert isdefined(@__MODULE__, :optimal_trajectory) && !isnothing(optimal_trajectory) "optimal_trajectory ist nicht initialisiert"
    end
    
    global logs = []
    global validation_scores
    global agent_save

    

    for j = 1:outer_loops

        println("outer loop $j")

        for i in 1:optimal_trainings
            RL.update_IL(agent.policy, optimal_trajectory)
        end


        if only_wind_steps > 0
            println("")
            println("Starting only wind learning...")
            stop_condition = StopAfterEpisodeWithMinSteps(only_wind_steps)

            global grid_price
            grid_price = ones(size(grid_price))

            # run start
            hook(PRE_EXPERIMENT_STAGE, agent, env)
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)
                hook(PRE_EPISODE_STAGE, agent, env)



                while !(is_terminated(env) || is_truncated(env))
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:], coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1

                    if stop_condition(agent, env)
                        is_stop = true
                        break
                    end
                end # end of an episode

                if is_terminated(env) || is_truncated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                    hook(POST_EPISODE_STAGE, agent, env)
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
        end


        for i = 1:inner_loops
            println("")
            stop_condition = StopAfterEpisodeWithMinSteps(num_steps)


            # run start
            hook(PRE_EXPERIMENT_STAGE, agent, env)
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)

                if same_day
                    env.y0 = generate_random_init(; same_day = true)
                    env.y = deepcopy(env.y0)

                    env.state = env.featurize(; env = env)
                else
                    hook(PRE_EPISODE_STAGE, agent, env)
                end

                while !(is_terminated(env) || is_truncated(env))
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action; reward_shaping = reward_shaping)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:], coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1

                    if stop_condition(agent, env)
                        is_stop = true
                        break
                    end
                end # end of an episode

                if is_terminated(env) || is_truncated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                    hook(POST_EPISODE_STAGE, agent, env)

                    if json
                        push!(logs, Dict(
                            "episode" => length(hook.rewards),
                            "reward_ep" => hook.rewards[end],
                            "actor_loss" => agent.policy.last_actor_loss,
                            "critic1_loss" => agent.policy.last_critic1_loss,
                            "critic2_loss" => agent.policy.last_critic2_loss,
                            "log_alpha" => agent.policy.log_α[1],
                            "q1_mean" => agent.policy.last_q1_mean,
                            "q2_mean" => agent.policy.last_q2_mean,
                            "target_q_mean" => agent.policy.last_target_q_mean,
                            "mean_minus_log_pi" => agent.policy.last_mean_minus_log_pi,
                        ))
                    end
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
            # run end


            println(hook.bestreward)

            if !same_day
                if @isdefined(validate_agent)
                    current_score = mean(validate_agent())

                    if !isempty(validation_scores) && current_score > maximum(validation_scores)
                        agent_save = deepcopy(agent)
                    end
                    
                    push!(validation_scores, current_score)
                end

                if !isempty(validation_scores)
                    println(lineplot(validation_scores, title="Validation scores", xlabel="Episode", ylabel="Score", color=:cyan))

                    println("Best validation score: $(maximum(validation_scores))")
                end
            end

            # hook.rewards = clamp.(hook.rewards, -3000, 0)

            
        end

        if plot_runs
            p1 = render_run(; exploration = true, new_day = !same_day)#, plot_values = true)
            #p2 = plot_critic(; return_plot = true)
            #display([p1 p2])
            #display(p1)
        end

    end

    if visuals && false
        rm(dirpath * "/training.mp4", force=true)
        run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
    end

    #save()
end


#train()
#train(;num_steps = 140)
#train(;visuals = true, num_steps = 70)


function load_agent(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save_agent(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end



function optimize_day(
    steps = 3000;
    verbose = true,
    prefer_early = true,
    prefer_early_mode = :weighted,
    prefer_early_weight = 1e-1,
    primary_reward_slack = 1e-6,
    ipopt_tol = 1e-6,
    ipopt_dual_inf_tol = 1e-3,
    ipopt_acceptable_tol = 1e-4,
    ipopt_acceptable_dual_inf_tol = 1e-2,
    ipopt_acceptable_constr_viol_tol = 1e-4,
    ipopt_acceptable_obj_change_tol = 1e-5,
    ipopt_acceptable_iter = 5,
    ipopt_nlp_scaling_method = "gradient-based",
    ipopt_hessian_approximation = "limited-memory",
)
    horizon = Int(te / dt)
    MOI = MathOptInterface
    accepted_statuses = Set([MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
    if isdefined(MOI, :ALMOST_OPTIMAL)
        push!(accepted_statuses, getfield(MOI, :ALMOST_OPTIMAL))
    end
    if isdefined(MOI, :ALMOST_LOCALLY_SOLVED)
        push!(accepted_statuses, getfield(MOI, :ALMOST_LOCALLY_SOLVED))
    end

    function build_base_model(max_iter)
        model = Model(Ipopt.Optimizer)

        if !verbose
            set_silent(model)
        end

        set_optimizer_attribute(model, "max_iter", max_iter)

        set_optimizer_attribute(model, "check_derivatives_for_naninf", "yes")
        set_optimizer_attribute(model, "print_info_string", "yes")
        set_optimizer_attribute(model, "nlp_scaling_method", ipopt_nlp_scaling_method)
        set_optimizer_attribute(model, "hessian_approximation", ipopt_hessian_approximation)

        set_optimizer_attribute(model, "tol", ipopt_tol)
        set_optimizer_attribute(model, "dual_inf_tol", ipopt_dual_inf_tol)
        set_optimizer_attribute(model, "acceptable_tol", ipopt_acceptable_tol)
        set_optimizer_attribute(model, "acceptable_dual_inf_tol", ipopt_acceptable_dual_inf_tol)
        set_optimizer_attribute(model, "acceptable_constr_viol_tol", ipopt_acceptable_constr_viol_tol)
        set_optimizer_attribute(model, "acceptable_obj_change_tol", ipopt_acceptable_obj_change_tol)
        set_optimizer_attribute(model, "acceptable_iter", ipopt_acceptable_iter)

        set_optimizer_attribute(model, "linear_solver", "mumps")

        @variable(model, -1 <= x[1:n_windCORES, 1:horizon] <= 1)
        action = (x .+ 1) .* 0.5
        @constraint(model, sum(action) == 100.03)  # slightly relaxed to help with numerical issues

        return model, x, action
    end

    # Stage 1: maximize the original objective.
    model1, x1, action1 = build_base_model(steps)
    primary_objective = evaluate(action1)
    frontload_score1 = sum((horizon - t + 1) * action1[i, t] for i in 1:n_windCORES, t in 1:horizon) / (horizon * 100.0)
    if prefer_early && prefer_early_mode == :weighted
        @objective(model1, Max, primary_objective + prefer_early_weight * frontload_score1)
    else
        @objective(model1, Max, primary_objective)
    end
    optimize!(model1)

    status1 = termination_status(model1)
    if !(status1 in accepted_statuses) && verbose
        println("IPOPT stage 1 status: $(status1)")
    end

    if !has_values(model1)
        error("IPOPT stage 1 returned no values (status=$(status1)).")
    end

    x1_values = value.(x1)

    if !prefer_early || prefer_early_mode == :weighted
        return x1_values
    end

    best_primary_value = objective_value(model1)

    # Stage 2 (lexicographic): among near-optimal primary solutions, front-load work as much as possible.
    model2, x2, action2 = build_base_model(steps)
    set_start_value.(x2, x1_values)

    primary_objective_stage2 = evaluate(action2)
    @constraint(model2, primary_objective_stage2 >= best_primary_value - primary_reward_slack)

    @objective(
        model2,
        Max,
        sum((horizon - t + 1) * action2[i, t] for i in 1:n_windCORES, t in 1:horizon)
    )

    optimize!(model2)

    if !has_values(model2)
        if verbose
            println("IPOPT stage 2 returned no values -> fallback to stage 1 solution")
        end
        return x1_values
    end

    x2_values = value.(x2)
    status2 = termination_status(model2)

    action2_values = (x2_values .+ 1) .* 0.5
    load_sum_violation = abs(sum(action2_values) - 100.0)
    primary2_value = value(primary_objective_stage2)
    min_primary_value = best_primary_value - primary_reward_slack
    primary_constraint_violation = max(0.0, min_primary_value - primary2_value)

    feasible_stage2 = status2 in accepted_statuses &&
                      load_sum_violation <= 1e-4 &&
                      primary_constraint_violation <= 1e-6

    if feasible_stage2
        return x2_values
    else
        if verbose
            println("IPOPT stage 2 status: $(status2) -> fallback to stage 1 solution")
            println("stage2 load-sum violation: $(load_sum_violation)")
            println("stage2 primary-constraint violation: $(primary_constraint_violation)")
        end
        return x1_values
    end
end





# sum(actions) has to be 100

function evaluate(actions; collect_rewards = false, reward_shaping = false)
    step = 2

    reward_sum = 0.0
    global rewards = Float64[]

    for t in 1:Int(te/dt)

        reward, _ = calculate_day(actions[:,t], nothing, t-1; reward_shaping = reward_shaping)

        reward_sum += reward

        if collect_rewards
            push!(rewards, reward)
        end

        step += 1
    end

    if collect_rewards
        rewards
    else
        reward_sum
    end
end

# train(num_steps = 14300, inner_loops = 2, optimized_episodes = 20, outer_loops = 100)

function plot_rewards(smoothing = 30)
    to_plot = Float64[]
    for i in smoothing:length(hook.rewards)
        push!(to_plot, mean(hook.rewards[i+1-smoothing:i]))
    end

    p = plot(to_plot)
    display(p)
end
