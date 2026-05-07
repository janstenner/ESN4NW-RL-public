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
using Distributions
using UnicodePlots
using CircularArrayBuffers

n_turbines = 1
n_jobs = 3
max_slots = 5
JOB_SPAWN_PROB = 0.05

action_dim = n_turbines * n_jobs
history_steps = 5
include_history_steps = 1
include_gradients = 2

te = Inf
dt = 5 / 1440
t0 = 0.0
min_best_episode = 1
terminated_at_timeout = false

if !(@isdefined dist)
    dist = Normal
end

if !(@isdefined multi_agent)
    multi_agent = false
end

if !(dist == Normal || dist == Categorical)
    error("Unsupported dist=$(dist). Complex supports Normal and Categorical.")
end

n_hist_local = clamp(include_history_steps, 1, history_steps)
n_grad_local = clamp(include_gradients, 0, 2)
base_state_dim = 1 + n_hist_local + n_grad_local + 1 +
                 n_turbines * (n_hist_local + n_grad_local + 1)

if multi_agent
    # Per-job observation: global features + own-job fields + own-job allocations across turbines.
    state_dim = base_state_dim + 3 + n_turbines
else
    state_dim = base_state_dim + 3 * n_jobs + n_turbines * n_jobs
end

sim_space = multi_agent ? Space(fill(0..1, (state_dim, n_jobs))) : Space(fill(0..1, (state_dim,)))

if dist == Categorical
    if multi_agent
        # One categorical policy head per job-agent.
        n_discrete_actions = (max_slots + 1)^n_turbines
        actionspace = Space(fill(0..1, (n_discrete_actions,)))
        env_actionspace = Space(fill(1:n_discrete_actions, (1, n_jobs)))
    else
        n_discrete_actions = (max_slots + 1)^action_dim
        actionspace = Space(fill(0..1, (n_discrete_actions,)))
        env_actionspace = Space(fill(1:n_discrete_actions, (1,)))
    end
else
    n_discrete_actions = 0
    actionspace = Space(fill(-1..1, (action_dim,)))
    env_actionspace = actionspace
end

mutable struct Job
    id::Int
    load::Int
    remaining::Int
    arrival_time::Float64
    deadline::Float64
    penalty::Float64
end

job_slots::Vector{Union{Job, Nothing}} = [nothing for _ in 1:n_jobs]
allocation::Matrix{Int} = zeros(Int, n_turbines, n_jobs)

wind_model_vars = rand(n_turbines, 5)
grid_model_vars = rand(5)
curtailment_threshold = 0.4
wind = Float64[]
grid_price = 0.0
grid_price_history = Float64[]
wind_history = zeros(Float64, n_turbines, history_steps)

# State normalization constants for job-related features.
job_remaining_scale = 80.0
job_time_left_scale = 0.5
job_penalty_scale = 4.0

env = nothing
id_counter = 1

function generate_wind()
    global wind_model_vars
    temp_wind = Float64[]

    if isnothing(env)
        time = 0.0
    else
        time = env.time
    end

    for i in 1:n_turbines
        t_mod = mod(time, 2pi)

        base_wind = 0.5 + 0.5 * sin(t_mod + wind_model_vars[i, 1] * 4pi)
        base_wind += (wind_model_vars[i, 2] * 0.3) * sin(4.5 * time + 1.0)
        base_wind += (wind_model_vars[i, 3] * 0.2) * sin(6.2 * time + 1.2)
        base_wind += (wind_model_vars[i, 4] * 0.2) * sin(8.3 * time + 1.7)

        wind_val = base_wind + 0.8 * (wind_model_vars[i, 5] - 0.3)
        wind_val = 0.5 + (wind_val - 0.3) * 0.7

        for j in eachindex(wind_model_vars[i, :])
            wind_model_vars[i, j] = clamp(0.8 * wind_model_vars[i, j] + 0.2 * sin(j * time), 0.0, 1.0)
        end

        wind_val = clamp(wind_val, 0.0, 1.0)
        wind_val = -1 * wind_val + 1.0
        push!(temp_wind, wind_val)
    end

    return temp_wind
end

function generate_grid_price()
    global grid_model_vars

    if isnothing(env)
        time = 0.0
    else
        time = env.time
    end

    t_day = mod(time, 1.0)

    base_price = 0.5 + 0.5 * cos(2pi * t_day)
    scaled_base_price = (grid_model_vars[1] - 0.5) * 0.5 + ((grid_model_vars[2] * 0.5) + 0.8) * base_price
    scaled_base_price += (grid_model_vars[3] * 0.3) * sin(9.3 * time)
    scaled_base_price += (grid_model_vars[4] * 0.2) * sin(14.3 * time)
    price_val = scaled_base_price + 0.25 * (grid_model_vars[5] - 0.5)

    for i in eachindex(grid_model_vars)
        grid_model_vars[i] = clamp(0.8 * grid_model_vars[i] + 0.2 * sin(i * time), 0.0, 1.0)
    end

    price_val = 0.5 + (price_val - 0.3) * 0.7
    return clamp(price_val, 0.0, 1.0)
end

function generate_curtailment_threshold()
    return 0.4
end

function generate_job(current_time::Float64)::Job
    global id_counter
    new_id = id_counter
    id_counter += 1

    load = Int(max(floor(rand(Normal(40, 10))), 0.0))
    window = rand() * 0.4 + (Int(ceil(load / 5)) / 288)
    deadline = current_time + window
    penalty = max(rand(Normal(1.5, 0.3)), 0.0) + 0.7

    return Job(new_id, load, load, current_time, deadline, penalty)
end

function initialize_signal_history!()
    global grid_price_history, wind_history
    grid_price_history = fill(grid_price, history_steps)
    wind_history = hcat([fill(wind[i], history_steps) for i in 1:n_turbines]...)'
    return nothing
end

function push_signal_history!()
    global grid_price_history, wind_history

    push!(grid_price_history, grid_price)
    if length(grid_price_history) > history_steps
        popfirst!(grid_price_history)
    end

    wind_history[:, 1:end-1] = wind_history[:, 2:end]
    wind_history[:, end] = wind
    return nothing
end

function reset_env_state!(; randomize_model = true)
    global job_slots, allocation, wind_model_vars, grid_model_vars, wind, grid_price, curtailment_threshold

    job_slots = [nothing for _ in 1:n_jobs]
    allocation = zeros(Int, n_turbines, n_jobs)

    if randomize_model
        wind_model_vars = rand(n_turbines, 5)
        grid_model_vars = rand(5)
    end

    wind = generate_wind()
    grid_price = generate_grid_price()
    curtailment_threshold = generate_curtailment_threshold()
    initialize_signal_history!()

    return nothing
end

function state_layout_indices()
    n_hist = clamp(include_history_steps, 1, history_steps)
    n_grad = clamp(include_gradients, 0, 2)

    idx = 1
    layout = Dict{Symbol, Any}()

    layout[:time] = idx
    idx += 1

    layout[:grid_hist] = collect(idx:(idx + n_hist - 1))
    idx += n_hist

    layout[:grid_grad] = collect(idx:(idx + n_grad - 1))
    idx += n_grad

    layout[:curtailment] = idx
    idx += 1

    wind_layout = Vector{Dict{Symbol, Any}}(undef, n_turbines)
    for t in 1:n_turbines
        w = Dict{Symbol, Any}()
        w[:hist] = collect(idx:(idx + n_hist - 1))
        idx += n_hist
        w[:grad] = collect(idx:(idx + n_grad - 1))
        idx += n_grad
        w[:curtail_excess] = idx
        idx += 1
        wind_layout[t] = w
    end
    layout[:wind] = wind_layout

    if multi_agent
        layout[:job] = Dict{Symbol, Int}(
            :remaining => idx,
            :time_left => idx + 1,
            :penalty => idx + 2,
        )
        idx += 3
        layout[:allocation_local] = collect(idx:(idx + n_turbines - 1))
        idx += n_turbines
    else
        job_layout = Vector{Dict{Symbol, Int}}(undef, n_jobs)
        for j in 1:n_jobs
            job_layout[j] = Dict{Symbol, Int}(
                :remaining => idx,
                :time_left => idx + 1,
                :penalty => idx + 2,
            )
            idx += 3
        end
        layout[:jobs] = job_layout

        layout[:allocation] = reshape(collect(idx:(idx + n_turbines * n_jobs - 1)), n_turbines, n_jobs)
        idx += n_turbines * n_jobs
    end

    layout[:state_dim] = idx - 1
    return layout
end

normalize01(value, scale) = Float32(clamp(value / scale, 0.0, 1.0))

function create_state(; env = nothing, step = 0, generate_day = true, same_day = false)
    global grid_price_history, wind_history, curtailment_threshold, job_slots, allocation

    if isnothing(env)
        if generate_day
            reset_env_state!(; randomize_model = !same_day)
        end
        abs_time = 0.0
    else
        abs_time = env.time + env.dt
    end

    time_of_day = mod(abs_time, 1.0)
    n_hist = clamp(include_history_steps, 1, history_steps)
    n_grad = clamp(include_gradients, 0, 2)
    gradient_dt = dt * 1440.0
    function append_global_features!(target::Vector{Float32})
        push!(target, Float32(time_of_day))
        for i in history_steps:-1:(history_steps - n_hist + 1)
            push!(target, Float32(grid_price_history[i]))
        end
        if n_grad > 0
            g1 = 500 * (grid_price_history[end] - grid_price_history[end - 1]) / gradient_dt
            push!(target, Float32(g1))
            if n_grad > 1
                g2 = 50_000 * (grid_price_history[end] - 2 * grid_price_history[end - 1] + grid_price_history[end - 2]) / (gradient_dt^2)
                push!(target, Float32(g2))
            end
        end
        push!(target, Float32(curtailment_threshold))
        for t in 1:n_turbines
            for i in history_steps:-1:(history_steps - n_hist + 1)
                push!(target, Float32(wind_history[t, i]))
            end
            if n_grad > 0
                wg1 = 500 * (wind_history[t, end] - wind_history[t, end - 1]) / gradient_dt
                push!(target, Float32(wg1))
                if n_grad > 1
                    wg2 = 50_000 * (wind_history[t, end] - 2 * wind_history[t, end - 1] + wind_history[t, end - 2]) / (gradient_dt^2)
                    push!(target, Float32(wg2))
                end
            end
            push!(target, Float32(max(0.0, wind_history[t, end] - curtailment_threshold)))
        end
        return nothing
    end

    if multi_agent
        y_cols = Vector{Vector{Float32}}(undef, n_jobs)
        for j in 1:n_jobs
            yy = Float32[]
            append_global_features!(yy)

            job = job_slots[j]
            if job !== nothing
                push!(yy, normalize01(job.remaining, job_remaining_scale))
                push!(yy, normalize01(max(0.0, job.deadline - abs_time), job_time_left_scale))
                push!(yy, normalize01(job.penalty, job_penalty_scale))
            else
                push!(yy, 0.0f0)
                push!(yy, 0.0f0)
                push!(yy, 0.0f0)
            end

            for i in 1:n_turbines
                push!(yy, Float32(allocation[i, j]))
            end

            @assert length(yy) == state_dim "State length mismatch (job $(j)): got $(length(yy)), expected $(state_dim)"
            y_cols[j] = yy
        end
        y = hcat(y_cols...)
        @assert size(y) == (state_dim, n_jobs) "State matrix mismatch: got $(size(y)), expected ($(state_dim), $(n_jobs))"
        return y
    else
        y = Float32[]
        append_global_features!(y)

        for job in job_slots
            if job !== nothing
                push!(y, normalize01(job.remaining, job_remaining_scale))
                push!(y, normalize01(max(0.0, job.deadline - abs_time), job_time_left_scale))
                push!(y, normalize01(job.penalty, job_penalty_scale))
            else
                push!(y, 0.0f0)
                push!(y, 0.0f0)
                push!(y, 0.0f0)
            end
        end

        for i in 1:size(allocation, 1)
            for j in 1:size(allocation, 2)
                push!(y, Float32(allocation[i, j]))
            end
        end

        @assert length(y) == state_dim "State length mismatch: got $(length(y)), expected $(state_dim)"
        @assert length(y) == state_layout_indices()[:state_dim] "State layout mismatch: got $(length(y)), expected $(state_layout_indices()[:state_dim])"
        return y
    end
end

reset_env_state!(; randomize_model = true)
y0 = create_state(; generate_day = false)

function calculate_day(action, env, step = nothing; reward_shaping = false)
    global wind, grid_price, curtailment_threshold, job_slots, allocation, n_turbines, n_jobs

    if isnothing(step)
        step = isnothing(env) ? 0 : env.steps
    end
    current_time = isnothing(env) ? 0.0 : env.time

    for i in 1:n_turbines
        for j in 1:n_jobs
            if job_slots[j] !== nothing
                allocation[i, j] = action[i, j]
            else
                allocation[i, j] = 0
            end
        end

        total_alloc = sum(allocation[i, :])
        if total_alloc > max_slots
            scale = max_slots / total_alloc
            scaled_alloc = [allocation[i, j] * scale for j in 1:n_jobs]
            new_alloc = [round(Int, x) for x in scaled_alloc]

            while sum(new_alloc) > max_slots
                idx = argmax(new_alloc)
                new_alloc[idx] -= 1
            end
            while sum(new_alloc) < max_slots
                idx = argmin(new_alloc)
                new_alloc[idx] += 1
            end

            for j in 1:n_jobs
                allocation[i, j] = new_alloc[j]
            end
        end
    end

    total_cost = 0.0
    job_reward = 0.0

    for i in 1:n_turbines
        allocated = sum(allocation[i, :])
        if allocated == 0
            allocated = 1
        end

        effective_alloc = allocated / max_slots
        free_energy = max(0.0, wind[i] - curtailment_threshold)
        grid_energy = max(0.0, effective_alloc - free_energy)
        cost = grid_energy * grid_price
        total_cost += cost

        for j in 1:n_jobs
            slots_for_job = allocation[i, j]
            if slots_for_job > 0 && (job_slots[j] !== nothing)
                job = job_slots[j]
                job.remaining -= slots_for_job
                if job.remaining <= 0
                    job_reward += job.penalty
                    job_slots[j] = nothing
                    for k in 1:n_turbines
                        allocation[k, j] = 0
                    end
                end
            end
        end
    end

    wind = generate_wind()
    grid_price = generate_grid_price()
    curtailment_threshold = generate_curtailment_threshold()
    push_signal_history!()

    for j in 1:n_jobs
        if job_slots[j] !== nothing
            job = job_slots[j]
            if current_time > job.deadline && job.remaining > 0
                job_reward -= job.penalty
                job_slots[j] = nothing
                for i in 1:n_turbines
                    allocation[i, j] = 0
                end
            end
        end
    end

    for j in 1:n_jobs
        if job_slots[j] === nothing && rand() < JOB_SPAWN_PROB
            job_slots[j] = generate_job(current_time)
        end
    end

    step_reward = -total_cost + job_reward
    return step_reward
end

function do_step(env)
    step_reward = calculate_day(env.p, env, env.steps)
    env.reward = multi_agent ? fill(step_reward, n_jobs) : [step_reward]
    y = create_state(; env = env, step = env.steps + 1, generate_day = false)
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

    if ndims(y) == 2
        return y
    end
    return reshape(y, length(y), 1)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        action = action0
    else
        action = env.action
    end

    if dist == Categorical
        base = max_slots + 1
        if multi_agent
            action_vec = if action isa Integer
                fill(Int(action), n_jobs)
            elseif action isa AbstractArray
                vec(Int.(round.(action)))
            else
                fill(Int(round(action)), n_jobs)
            end
            if length(action_vec) == 1 && n_jobs > 1
                action_vec = fill(action_vec[1], n_jobs)
            end
            @assert length(action_vec) == n_jobs "Expected $(n_jobs) categorical actions, got $(length(action_vec))."

            action_vec = clamp.(action_vec, 1, n_discrete_actions)
            decoded = zeros(Int, n_turbines, n_jobs)
            for j in 1:n_jobs
                idx0 = action_vec[j] - 1
                for i in 1:n_turbines
                    decoded[i, j] = idx0 % base
                    idx0 ÷= base
                end
            end
            action = decoded
        else
            action_idx = if action isa Integer
                Int(action)
            elseif action isa AbstractArray
                Int(round(first(action)))
            else
                Int(round(action))
            end
            action_idx = clamp(action_idx, 1, n_discrete_actions)

            idx0 = action_idx - 1
            slots = zeros(Int, action_dim)
            for d in 1:action_dim
                slots[d] = idx0 % base
                idx0 ÷= base
            end
            action = reshape(slots, n_turbines, n_jobs)
        end
    else
        action = (action .+ 1) .* 0.5
        clamp!(action, 0.0, 1.0)
        action = action .* 5.0
        action = Int.(floor.(reshape(action, n_turbines, n_jobs)))
    end

    return action
end

function create_env()
    action0 = if dist == Categorical
        multi_agent ? ones(Int, 1, n_jobs) : 1
    else
        nothing
    end
    return GeneralEnv(
        do_step = do_step,
        reward_function = reward_function,
        featurize = featurize,
        prepare_action = prepare_action,
        y0 = y0,
        action0 = action0,
        te = te,
        t0 = t0,
        dt = dt,
        sim_space = sim_space,
        action_space = env_actionspace,
        max_value = 1.0,
        check_max_value = "nothing",
        terminated_at_timeout = terminated_at_timeout,
    )
end

function render_run(steps = 864; make_deterministic = true)
    global results = Dict("rewards" => [], "grid_price" => [])
    layout = state_layout_indices()

    for k in 1:n_turbines
        results["wind$(k)"] = []
    end

    for k in 1:n_jobs
        results["job$(k)_remaining"] = []
        results["job$(k)_remaining_50"] = []
        results["job$(k)_compute"] = []
        results["job$(k)_penalty"] = []
        results["job$(k)_time_left"] = []
    end

    slot_rows = n_turbines * max_slots
    slot_history = zeros(Int, slot_rows, steps)

    function fill_slot_column!(col::AbstractVector{Int}, alloc_mat)
        fill!(col, 0)
        for turb in 1:n_turbines
            base_row = (turb - 1) * max_slots
            slot_pos = 1
            for job in 1:n_jobs
                slot_pos > max_slots && break
                n_fill = clamp(Int(round(alloc_mat[turb, job])), 0, max_slots - slot_pos + 1)
                for _ in 1:n_fill
                    col[base_row + slot_pos] = job
                    slot_pos += 1
                    slot_pos > max_slots && break
                end
            end
        end
        return nothing
    end

    reset!(env)
    generate_random_init()

    env.time = rand() * 10000

    reward_sum = 0.0

    get_state_value(idx::Int, job::Int=1) = ndims(env.y) == 2 ? env.y[idx, job] : env.y[idx]

    for step_idx in 1:steps
        action = if make_deterministic
            if dist == Categorical
                action_dist = prob(agent.policy, env)
                if multi_agent
                    reshape(Int[mode(action_dist[j]) for j in 1:length(action_dist)], 1, n_jobs)
                else
                    @assert length(action_dist) == 1 "Categorical Complex expects one discrete action head."
                    mode(action_dist[1])
                end
            else
                prob(agent.policy, env).μ
            end
        else
            agent(env)
        end
        env(action)

        fill_slot_column!(view(slot_history, :, step_idx), allocation)

        for k in 1:n_turbines
            push!(results["wind$(k)"], get_state_value(layout[:wind][k][:hist][1], 1))
        end

        for k in 1:n_jobs
            if multi_agent
                job_idx = layout[:job]
                push!(results["job$(k)_remaining"], get_state_value(job_idx[:remaining], k))
                push!(results["job$(k)_time_left"], get_state_value(job_idx[:time_left], k))
                push!(results["job$(k)_penalty"], get_state_value(job_idx[:penalty], k))
            else
                job_idx = layout[:jobs][k]
                push!(results["job$(k)_remaining"], get_state_value(job_idx[:remaining], 1))
                push!(results["job$(k)_time_left"], get_state_value(job_idx[:time_left], 1))
                push!(results["job$(k)_penalty"], get_state_value(job_idx[:penalty], 1))
            end
            rem50 = job_slots[k] === nothing ? 0.0 : Float64(job_slots[k].remaining) / 50.0
            push!(results["job$(k)_remaining_50"], rem50)
            push!(results["job$(k)_compute"], sum(env.p[:, k]))
        end

        push!(results["rewards"], env.reward[1])
        push!(results["grid_price"], get_state_value(layout[:grid_hist][1], 1))
        reward_sum += mean(env.reward)
    end

    println(reward_sum)

    p = make_subplots(rows = 4 + n_jobs, cols = 1)

    area_colors = [
        "rgba(239, 83, 80, 0.30)",
        "rgba(66, 165, 245, 0.30)",
        "rgba(102, 187, 106, 0.30)",
        "rgba(255, 202, 40, 0.30)",
        "rgba(171, 71, 188, 0.30)",
        "rgba(38, 198, 218, 0.30)",
    ]
    line_colors = [
        "rgba(239, 83, 80, 0.95)",
        "rgba(66, 165, 245, 0.95)",
        "rgba(102, 187, 106, 0.95)",
        "rgba(255, 202, 40, 0.95)",
        "rgba(171, 71, 188, 0.95)",
        "rgba(38, 198, 218, 0.95)",
    ]

    for k in 1:n_jobs
        cidx = mod1(k, length(area_colors))
        add_trace!(
            p,
            scatter(
                y = results["job$(k)_remaining_50"],
                name = "job$(k) remaining/50",
                mode = "lines",
                line = attr(color = line_colors[cidx], width = 1),
                fill = "tozeroy",
                fillcolor = area_colors[cidx],
            ),
            row = 1,
            col = 1,
        )
    end

    add_trace!(p, scatter(y = results["rewards"], name = "reward", yaxis = "y2"), row = 2)
    add_trace!(p, scatter(y = results["grid_price"], name = "grid price"), row = 3, col = 1)

    for k in 1:n_turbines
        add_trace!(p, scatter(y = results["wind$(k)"], name = "wind$(k)"), row = 3, col = 1)
    end

    for k in 1:n_jobs
        cidx = mod1(k, length(line_colors))
        add_trace!(
            p,
            scatter(
                y = results["job$(k)_compute"],
                name = "job$(k) compute",
                mode = "lines",
                line = attr(color = line_colors[cidx], width = 1.5),
            ),
            row = 3 + k,
            col = 1,
        )
    end

    slot_history_plot = similar(slot_history)
    slot_labels = String[]
    for turb in 1:n_turbines
        row_start = (turb - 1) * max_slots + 1
        row_end = turb * max_slots
        slot_history_plot[row_start:row_end, :] .= reverse(slot_history[row_start:row_end, :], dims=1)
        for s in max_slots:-1:1
            push!(slot_labels, "T$(turb)-S$(s)")
        end
    end

    slot_colorscale = Any[]
    n_bins = n_jobs + 1
    for c in 0:n_jobs
        ccol = c == 0 ? "rgba(0,0,0,1.0)" : line_colors[mod1(c, length(line_colors))]
        x0 = c / n_bins
        x1 = (c + 1) / n_bins
        push!(slot_colorscale, [x0, ccol])
        push!(slot_colorscale, [x1, ccol])
    end

    add_trace!(
        p,
        PlotlyJS.heatmap(
            z = slot_history_plot,
            y = slot_labels,
            zmin = 0,
            zmax = n_jobs,
            colorscale = slot_colorscale,
            showscale = false,
            hoverongaps = false,
        ),
        row = 4 + n_jobs,
        col = 1,
    )

    display(p)
end

function generate_random_init()
    global wind, grid_price, curtailment_threshold, job_slots, allocation
    reset_env_state!(; randomize_model = true)
    new_y0 = create_state(; generate_day = false)

    if @isdefined(env) && !isnothing(env)
        env.y0 = deepcopy(new_y0)
        env.y = deepcopy(new_y0)
        env.state = env.featurize(; env = env)
    end

    return new_y0
end

train_rewards = Float64[]
temp_reward_queue::CircularArrayBuffer{Float64} = CircularArrayBuffer{Float64}(1)

function train(use_random_init = true; num_steps = 5_000_000, smoothing_window = 4000, collect_every = 16000, plot_every = 20_000)
    frame = 1

    global train_rewards
    global temp_reward_queue

    temp_reward_queue = CircularArrayBuffer{Float64}(smoothing_window)

    if use_random_init
        hook.generate_random_init = generate_random_init
    else
        hook.generate_random_init = false
    end

    stop_condition = StopAfterStep(num_steps)

    hook(PRE_EXPERIMENT_STAGE, agent, env)
    agent(PRE_EXPERIMENT_STAGE, env)
    is_stop = false

    while !is_stop
        reset!(env)
        agent(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, agent, env)

        while !is_terminated(env)
            action = agent(env)

            agent(PRE_ACT_STAGE, env, action)
            hook(PRE_ACT_STAGE, agent, env, action)

            env(action)

            agent(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, agent, env)

            frame += 1
            push!(temp_reward_queue, env.reward[1])

            if frame > smoothing_window && frame % collect_every == 0
                push!(train_rewards, mean(temp_reward_queue))
            end

            if frame % plot_every == 0
                plt = lineplot(train_rewards, title = "Current smoothed rewards", xlabel = "Steps", ylabel = "Score")
                println(plt)
                render_run()
            end

            if stop_condition(agent, env)
                is_stop = true
                break
            end
        end

        if is_terminated(env)
            agent(POST_EPISODE_STAGE, env)
            hook(POST_EPISODE_STAGE, agent, env)
        end
    end

    hook(POST_EXPERIMENT_STAGE, agent, env)
end

function _save_dirpath()
    return isdefined(@__MODULE__, :dirpath) ? getfield(@__MODULE__, :dirpath) : @__DIR__
end

function load_agent(number = nothing)
    base = _save_dirpath()
    if isnothing(number)
        global hook = FileIO.load(base * "/saves/hook.jld2", "hook")
        global agent = FileIO.load(base * "/saves/agent.jld2", "agent")
        global train_rewards = FileIO.load(base * "/saves/train_rewards.jld2", "train_rewards")
    else
        global hook = FileIO.load(base * "/saves/hook$number.jld2", "hook")
        global agent = FileIO.load(base * "/saves/agent$number.jld2", "agent")
        global train_rewards = FileIO.load(base * "/saves/train_rewards$number.jld2", "train_rewards")
    end
end

function save_agent(number = nothing)
    base = _save_dirpath()
    isdir(base * "/saves") || mkdir(base * "/saves")

    if isnothing(number)
        FileIO.save(base * "/saves/hook.jld2", "hook", hook)
        FileIO.save(base * "/saves/agent.jld2", "agent", agent)
        FileIO.save(base * "/saves/train_rewards.jld2", "train_rewards", train_rewards)
    else
        FileIO.save(base * "/saves/hook$number.jld2", "hook", hook)
        FileIO.save(base * "/saves/agent$number.jld2", "agent", agent)
        FileIO.save(base * "/saves/train_rewards$number.jld2", "train_rewards", train_rewards)
    end
end

function plot_rewards(smoothing = 30)
    to_plot = Float64[]
    for i in smoothing:length(train_rewards)
        push!(to_plot, mean(train_rewards[i + 1 - smoothing:i]))
    end

    p = plot(to_plot)
    display(p)
end
