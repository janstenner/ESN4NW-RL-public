
validation_set_file = joinpath(@__DIR__, "validation_set.jld2")
validation_results_file = joinpath(@__DIR__, "validation_results.jld2")

function load_validation_set(file_path::String)
    loaded = FileIO.load(file_path)
    if haskey(loaded, "validation_set")
        return loaded["validation_set"]
    elseif haskey(loaded, "set")
        validation_set = loaded["set"]
        FileIO.save(file_path, "validation_set", validation_set)
        return validation_set
    else
        error("No validation set found in $file_path")
    end
end

global validation_set = load_validation_set(validation_set_file)
# FileIO.save(validation_set_file, "validation_set", validation_set)

global validation_results = FileIO.load(validation_results_file, "validation_results")
# FileIO.save(validation_results_file, "validation_results", validation_results)


function plot_validation_results(; current = false)
    traces = AbstractTrace[]

    for (key, value) in validation_results
        trace = box(y=value, name=key, boxpoints="all", quartilemethod="linear")
        push!(traces, trace)
    end

    global agent_save
    if current && !isnothing(agent_save)
        global agent
        agent_temp = deepcopy(agent)
        agent = deepcopy(agent_save)

        current_scores = validate_agent()

        trace = box(y=current_scores, name="Current Agent", boxpoints="all", quartilemethod="linear")
        push!(traces, trace)
        agent = deepcopy(agent_temp)
        
    elseif isnothing(agent_save)
        println("Current failed: agent_save is nothing!")
    end

    p = plot(traces)
    display(p)
end

# data = [1,2,3,4,5,6,7,8,9]
# trace1 = box(y=data, boxpoints="all", quartilemethod="linear", name="linear")
# trace2 = box(y=data, boxpoints="all", quartilemethod="inclusive", name="inclusive")
# trace3 = box(y=data, boxpoints="all", quartilemethod="exclusive", name="exclusive")
# plot([trace1, trace2, trace3])




function validate_agent(; optimizer = false, render = false)

    global validation_rewards = Float32[]
    xx = collect(dt/60:dt/60:te/60)

    for (w, gp) in validation_set

        reset!(env)
        # Perform validation checks on wind and gp
        global wind = [w]
        global grid_price = gp

        y0 = create_state(; generate_day = false)

        env.y0 = deepcopy(y0)
        env.y = deepcopy(y0)
        env.state = env.featurize(; env = env)

        global results_run = Dict("rewards" => [], "loadleft" => [], "hpc" => [])

        if optimizer
            optimal_actions = optimize_day(5000)
            reward_sum = Float32(sum(evaluate(optimal_actions; collect_rewards = true)))

        else
            reward_sum = 0.0f0
            while !(env.terminated || env.truncated)

                if hasproperty(agent.policy, :actor)
                    action = agent.policy.actor.μ(env.state)
                elseif hasproperty(agent.policy, :approximator)
                    action = agent.policy.approximator.actor.μ(env.state)
                elseif hasproperty(agent.policy, :behavior_actor)
                    action = agent.policy.behavior_actor(env.state)
                end
                
                env(action; reward_shaping = false)

                reward_sum += mean(env.reward)

                if render
                    push!(results_run["hpc"], clamp((action[1]+1)*0.5, 0, 1)) #env.p[k])
                    push!(results_run["rewards"], env.reward[1])
                    push!(results_run["loadleft"], env.y[1])
                end
                
            end
        end

        if render
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
                            ),#range=[0,1]),
                            yaxis2 = attr(
                                overlaying="y",
                                side="right",
                                titlefont_color="orange",
                                #range=[-1, 1]
                            ),
                        )
            to_plot = AbstractTrace[]
            push!(to_plot, scatter(x=xx, y=results_run["rewards"], name="Reward", yaxis = "y2"))
            push!(to_plot, scatter(x=xx, y=results_run["loadleft"], name="Load Left"))
            push!(to_plot, scatter(x=xx, y=grid_price[history_steps:end], name="Grid Price"))
            push!(to_plot, scatter(x=xx, y=results_run["hpc"], name="WindCORE utilization"))
            push!(to_plot, scatter(x=xx, y=wind[1][history_steps:end], name="Wind Power"))
            plott = plot(Vector(to_plot), layout)
            display(plott)
        end

        push!(validation_rewards, reward_sum)
    end

    validation_rewards
end




function generate_validation_set(;n = 200)

    global validation_set = []

    for i in 1:n
        wind = generate_wind()
        gp = generate_grid_price()
        push!(validation_set, (wind, gp))
    end

    return validation_set
end




