Base.@kwdef mutable struct GeneralHook <: AbstractHook

    rewards::Vector{Float64} = Float64[]
    rewards_compare::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
    rewards_all_timesteps::Vector{Float64} = Float64[]
    ep = 1

    is_display_on_exit::Bool = true
    display_after_episode = false
    generate_random_init = nothing
    collect_history::Bool = false
    collect_NNA::Bool = false
    collect_bestDF::Bool = false
    collect_rewards_all_timesteps::Bool = true

    min_best_episode = 0
    early_success_possible = false
    bestNNA = nothing
    bestDF = DataFrame()
    bestreward = -1000000.0
    bestepisode = 0
    currentNNA = nothing
    currentDF = DataFrame()
    history = []
    errored_episodes = []
    error_detection = function(y) return false end
end

Base.getindex(h::GeneralHook) = h.rewards

function (hook::GeneralHook)(::PreExperimentStage, agent, env)
    if hook.collect_NNA && hook.currentNNA === nothing
        hook.currentNNA = deepcopy(agent.policy.behavior_actor)
        hook.bestNNA = deepcopy(agent.policy.behavior_actor)
    end
end

function (hook::GeneralHook)(::PreEpisodeStage, agent, env)
    if !(isnothing(hook.generate_random_init))
        env.y0 = hook.generate_random_init()
        env.y = deepcopy(env.y0)

        env.state = env.featurize(; env = env)
    end
end

function (hook::GeneralHook)(::PreActStage, agent, env, action)

end

function (hook::GeneralHook)(::PostActStage, agent, env)
    hook.reward += mean(reward(env))
    if hook.collect_rewards_all_timesteps
        push!(hook.rewards_all_timesteps, mean(reward(env)))
    end

    if hook.collect_bestDF
        tmp = DataFrame()
        insertcols!(tmp, :timestep => env.steps)
        insertcols!(tmp, :action => [deepcopy(vec(env.action))])
        insertcols!(tmp, :p => [deepcopy(send_to_host(env.p))])
        insertcols!(tmp, :y => [deepcopy(send_to_host(env.y))])
        insertcols!(tmp, :reward => [reward(env)])
        append!(hook.currentDF, tmp)
    end
end

function (hook::GeneralHook)(::PostEpisodeStage, agent, env)
    end_successful = false
    if hook.early_success_possible
        end_successful = hook.ep >= hook.min_best_episode ? true : false
    else
        end_successful = (env.time >= env.te && hook.ep >= hook.min_best_episode) ? true : false
    end

    if end_successful
        push!(hook.rewards_compare, hook.reward)
        if length(hook.rewards_compare) >= 1 && hook.reward >= maximum(hook.rewards_compare)
            hook.bestreward = hook.reward
            hook.bestepisode = hook.ep

            if hook.collect_NNA 
                copyto!(hook.bestNNA, agent.policy.behavior_actor)
            end

            if hook.collect_bestDF
                hook.bestDF = copy(hook.currentDF)
            end
        end
    end

    if env.time < env.te
        if hook.error_detection(env.y)
            push!(hook.errored_episodes, hook.ep)
        end
    end
    
    if hook.collect_history
        push!(hook.history, hook.currentDF)
    end
    hook.currentDF = DataFrame()

    hook.ep += 1

    push!(hook.rewards, hook.reward)
    hook.reward = 0
    
    if hook.collect_NNA
        copyto!(hook.currentNNA, agent.policy.behavior_actor)
    end

    if hook.display_after_episode && !isempty(hook.rewards)
        println(lineplot(hook.rewards, title="Total reward per episode", xlabel="Episode", ylabel="Score"))
    end
end

function (hook::GeneralHook)(::PostExperimentStage, agent, env)
    if hook.is_display_on_exit && !isempty(hook.rewards)
        println(lineplot(hook.rewards, title="Total reward per episode", xlabel="Episode", ylabel="Score"))
    end
end