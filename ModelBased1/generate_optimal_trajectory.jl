using FileIO, JLD2
using RL

# File to store optimal trajectories
trajectories_file = joinpath(@__DIR__, "optimal_trajectories.jld2")

optimal_episodes = 1_000

const TRAJECTORY_ALGORITHMS = ["SAC", "SAC2", "DDPG", "PPO", "PPO2", "PPO3", "FlowPPO"]
const TRAJECTORY_MODES = ("no_RS", "with_RS")
const TRAJECTORY_REQUIRED_KEYS = Dict(
    "SAC" => (:state, :action, :reward, :terminated, :truncated, :next_state),
    "SAC2" => (:state, :action, :reward, :terminated, :truncated, :next_state),
    "DDPG" => (:state, :action, :reward, :terminated, :truncated),
    "PPO" => (:state, :action, :action_log_prob, :reward, :terminated, :truncated, :next_state),
    "PPO2" => (:state, :action, :action_log_prob, :reward, :explore_mod, :terminated, :truncated, :next_state),
    "PPO3" => (:state, :action, :action_log_prob, :reward, :explore_mod, :terminated, :truncated, :next_state),
    "FlowPPO" => (:state, :action, :action_log_prob, :reward, :explore_mod, :terminated, :truncated, :next_state),
)

function initialize_trajectories_dict()
    dict = Dict{String, Dict{String, Any}}()
    for alg in TRAJECTORY_ALGORITHMS
        dict[alg] = Dict{String, Any}()
    end
    return dict
end

function ensure_alg_dict!(dict, alg::String)
    if !haskey(dict, alg)
        dict[alg] = Dict{String, Any}()
    end
    return nothing
end

function trajectory_has_keys(traj, required_keys)
    all(k -> (hasmethod(haskey, Tuple{typeof(traj), typeof(k)}) && haskey(traj, k)), required_keys)
end

function trajectories_up_to_date(dict)
    for alg in TRAJECTORY_ALGORITHMS
        haskey(dict, alg) || return false
        alg_dict = dict[alg]
        alg_dict isa AbstractDict || return false
        for mode in TRAJECTORY_MODES
            haskey(alg_dict, mode) || return false
            trajectory_has_keys(alg_dict[mode], TRAJECTORY_REQUIRED_KEYS[alg]) || return false
        end
    end
    return true
end

function create_sac_like_trajectory(capacity, state_dim, action_dim, n_envs)
    CircularArrayTrajectory(;
        capacity = capacity,
        state = Float32 => (state_dim, n_envs),
        action = Float32 => (action_dim, n_envs),
        reward = Float32 => (n_envs),
        terminated = Bool => (n_envs,),
        truncated = Bool => (n_envs,),
        next_state = Float32 => (state_dim, n_envs),
    )
end

function create_ppo_like_trajectory(capacity, state_dim, action_dim, n_envs; with_explore_mod = false)
    if with_explore_mod
        return CircularArrayTrajectory(;
            capacity = capacity,
            state = Float32 => (state_dim, n_envs),
            action = Float32 => (action_dim, n_envs),
            action_log_prob = Float32 => (n_envs),
            reward = Float32 => (n_envs),
            explore_mod = Float32 => (n_envs),
            terminated = Bool => (n_envs,),
            truncated = Bool => (n_envs,),
            next_state = Float32 => (state_dim, n_envs),
        )
    end

    return CircularArrayTrajectory(;
        capacity = capacity,
        state = Float32 => (state_dim, n_envs),
        action = Float32 => (action_dim, n_envs),
        action_log_prob = Float32 => (n_envs),
        reward = Float32 => (n_envs),
        terminated = Bool => (n_envs,),
        truncated = Bool => (n_envs,),
        next_state = Float32 => (state_dim, n_envs),
    )
end

function create_ddpg_trajectory(capacity, state_dim, action_dim)
    CircularArrayTrajectory(;
        capacity = capacity,
        state = Float32 => state_dim,
        action = Float32 => action_dim,
        reward = Float32 => 1,
        terminated = Bool => (),
        truncated = Bool => (),
    )
end

# Load or initialize trajectories dictionary
if isfile(trajectories_file)
    trajectories = FileIO.load(trajectories_file, "trajectories")
    println("Loaded existing trajectories file")
else
    trajectories = initialize_trajectories_dict()
    println("Created new trajectories dictionary")
end

function generate_optimal_trajectories(; steps = 10_000)
    global env, optimal_episodes, gamma

    n_envs = 1
    state_dim = size(env.state_space)[1]
    action_dim_local = size(env.action_space)[1]
    capacity = Int(te / dt) * optimal_episodes

    # SAC / SAC2
    sac_trajectory_no_rs = create_sac_like_trajectory(capacity, state_dim, action_dim_local, n_envs)
    sac_trajectory_rs = create_sac_like_trajectory(capacity, state_dim, action_dim_local, n_envs)
    sac2_trajectory_no_rs = create_sac_like_trajectory(capacity, state_dim, action_dim_local, n_envs)
    sac2_trajectory_rs = create_sac_like_trajectory(capacity, state_dim, action_dim_local, n_envs)

    # DDPG
    ddpg_trajectory_no_rs = create_ddpg_trajectory(capacity, state_dim, action_dim_local)
    ddpg_trajectory_rs = create_ddpg_trajectory(capacity, state_dim, action_dim_local)

    # PPO
    ppo_trajectory_no_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs)
    ppo_trajectory_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs)

    # PPO2 / PPO3 / FlowPPO
    ppo2_trajectory_no_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs; with_explore_mod = true)
    ppo2_trajectory_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs; with_explore_mod = true)
    ppo3_trajectory_no_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs; with_explore_mod = true)
    ppo3_trajectory_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs; with_explore_mod = true)
    flowppo_trajectory_no_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs; with_explore_mod = true)
    flowppo_trajectory_rs = create_ppo_like_trajectory(capacity, state_dim, action_dim_local, n_envs; with_explore_mod = true)

    global optimal_rewards = Float64[]

    for i in 1:optimal_episodes
        println("Optimized Episode $(i)...")
        reset!(env)
        generate_random_init()

        optimal_actions = optimize_day(steps; verbose = false)
        n = 1

        while !(is_terminated(env) || is_truncated(env))
            if n <= size(optimal_actions, 2)
                action = hcat(optimal_actions[:, n])
            else
                action = 0.001f0 .* ones(action_dim_local, 1)
            end

            state_before = env.state
            last_action_log_prob = ones(Float32, n_envs) .* 0.1
            explore_mod = ones(Float32, n_envs)

            # SAC / SAC2
            push!(sac_trajectory_no_rs; state = env.state, action = action)
            push!(sac_trajectory_rs; state = env.state, action = action)
            push!(sac2_trajectory_no_rs; state = env.state, action = action)
            push!(sac2_trajectory_rs; state = env.state, action = action)

            # DDPG (single-env traces)
            push!(ddpg_trajectory_no_rs; state = vec(state_before[:, 1]), action = vec(action[:, 1]))
            push!(ddpg_trajectory_rs; state = vec(state_before[:, 1]), action = vec(action[:, 1]))

            # PPO
            push!(ppo_trajectory_no_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
            )
            push!(ppo_trajectory_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
            )

            # PPO2 / PPO3 / FlowPPO
            push!(ppo2_trajectory_no_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
                explore_mod = explore_mod,
            )
            push!(ppo2_trajectory_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
                explore_mod = explore_mod,
            )
            push!(ppo3_trajectory_no_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
                explore_mod = explore_mod,
            )
            push!(ppo3_trajectory_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
                explore_mod = explore_mod,
            )
            push!(flowppo_trajectory_no_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
                explore_mod = explore_mod,
            )
            push!(flowppo_trajectory_rs;
                state = env.state,
                action = action,
                action_log_prob = last_action_log_prob,
                explore_mod = explore_mod,
            )

            compute_left_before = env.y[1]
            env(action; reward_shaping = false)

            r = env.reward[1]
            compute_left_after = env.y[1]

            beta = 1.0
            r_shaped = r + beta * (compute_left_before - compute_left_after - (gamma - 1) * compute_left_after)
            r_shaped *= reward_scale_factor

            term = is_terminated(env)
            trunc = is_truncated(env)
            done = term || trunc

            # SAC / SAC2
            push!(sac_trajectory_no_rs[:reward], [r])
            push!(sac_trajectory_rs[:reward], [r_shaped])
            push!(sac_trajectory_no_rs[:terminated], term)
            push!(sac_trajectory_rs[:terminated], term)
            push!(sac_trajectory_no_rs[:truncated], trunc)
            push!(sac_trajectory_rs[:truncated], trunc)
            push!(sac_trajectory_no_rs[:next_state], env.state)
            push!(sac_trajectory_rs[:next_state], env.state)

            push!(sac2_trajectory_no_rs[:reward], [r])
            push!(sac2_trajectory_rs[:reward], [r_shaped])
            push!(sac2_trajectory_no_rs[:terminated], term)
            push!(sac2_trajectory_rs[:terminated], term)
            push!(sac2_trajectory_no_rs[:truncated], trunc)
            push!(sac2_trajectory_rs[:truncated], trunc)
            push!(sac2_trajectory_no_rs[:next_state], env.state)
            push!(sac2_trajectory_rs[:next_state], env.state)

            # DDPG
            push!(ddpg_trajectory_no_rs[:reward], Float32(r))
            push!(ddpg_trajectory_rs[:reward], Float32(r_shaped))
            push!(ddpg_trajectory_no_rs[:terminated], done)
            push!(ddpg_trajectory_rs[:terminated], done)
            push!(ddpg_trajectory_no_rs[:truncated], trunc)
            push!(ddpg_trajectory_rs[:truncated], trunc)

            # PPO
            push!(ppo_trajectory_no_rs[:reward], [r])
            push!(ppo_trajectory_rs[:reward], [r_shaped])
            push!(ppo_trajectory_no_rs[:terminated], term)
            push!(ppo_trajectory_rs[:terminated], term)
            push!(ppo_trajectory_no_rs[:truncated], trunc)
            push!(ppo_trajectory_rs[:truncated], trunc)
            push!(ppo_trajectory_no_rs[:next_state], env.state)
            push!(ppo_trajectory_rs[:next_state], env.state)

            # PPO2 / PPO3 / FlowPPO
            push!(ppo2_trajectory_no_rs[:reward], [r])
            push!(ppo2_trajectory_rs[:reward], [r_shaped])
            push!(ppo2_trajectory_no_rs[:terminated], term)
            push!(ppo2_trajectory_rs[:terminated], term)
            push!(ppo2_trajectory_no_rs[:truncated], trunc)
            push!(ppo2_trajectory_rs[:truncated], trunc)
            push!(ppo2_trajectory_no_rs[:next_state], env.state)
            push!(ppo2_trajectory_rs[:next_state], env.state)

            push!(ppo3_trajectory_no_rs[:reward], [r])
            push!(ppo3_trajectory_rs[:reward], [r_shaped])
            push!(ppo3_trajectory_no_rs[:terminated], term)
            push!(ppo3_trajectory_rs[:terminated], term)
            push!(ppo3_trajectory_no_rs[:truncated], trunc)
            push!(ppo3_trajectory_rs[:truncated], trunc)
            push!(ppo3_trajectory_no_rs[:next_state], env.state)
            push!(ppo3_trajectory_rs[:next_state], env.state)

            push!(flowppo_trajectory_no_rs[:reward], [r])
            push!(flowppo_trajectory_rs[:reward], [r_shaped])
            push!(flowppo_trajectory_no_rs[:terminated], term)
            push!(flowppo_trajectory_rs[:terminated], term)
            push!(flowppo_trajectory_no_rs[:truncated], trunc)
            push!(flowppo_trajectory_rs[:truncated], trunc)
            push!(flowppo_trajectory_no_rs[:next_state], env.state)
            push!(flowppo_trajectory_rs[:next_state], env.state)

            n += 1
        end
    end

    for alg in TRAJECTORY_ALGORITHMS
        ensure_alg_dict!(trajectories, alg)
    end

    trajectories["SAC"]["no_RS"] = sac_trajectory_no_rs
    trajectories["SAC"]["with_RS"] = sac_trajectory_rs
    trajectories["SAC2"]["no_RS"] = sac2_trajectory_no_rs
    trajectories["SAC2"]["with_RS"] = sac2_trajectory_rs
    trajectories["DDPG"]["no_RS"] = ddpg_trajectory_no_rs
    trajectories["DDPG"]["with_RS"] = ddpg_trajectory_rs
    trajectories["PPO"]["no_RS"] = ppo_trajectory_no_rs
    trajectories["PPO"]["with_RS"] = ppo_trajectory_rs
    trajectories["PPO2"]["no_RS"] = ppo2_trajectory_no_rs
    trajectories["PPO2"]["with_RS"] = ppo2_trajectory_rs
    trajectories["PPO3"]["no_RS"] = ppo3_trajectory_no_rs
    trajectories["PPO3"]["with_RS"] = ppo3_trajectory_rs
    trajectories["FlowPPO"]["no_RS"] = flowppo_trajectory_no_rs
    trajectories["FlowPPO"]["with_RS"] = flowppo_trajectory_rs

    FileIO.save(trajectories_file, "trajectories", trajectories)
    println("\nAll optimal trajectories generated and saved to $trajectories_file")
end

# Generate trajectories if they don't exist or if an old structure is loaded.
if !trajectories_up_to_date(trajectories)
    println("Regenerating trajectories to match current RL agent schemas...")
    generate_optimal_trajectories()
end
