
function _normalize_validation_entry(entry)
    if entry isa Tuple && length(entry) == 3
        w, gp, ct = entry
        return (w, gp, ct)
    elseif entry isa Tuple && length(entry) == 2
        # Backward compatibility for older validation_set schema.
        w, gp = entry
        ct = fill(curtailment_threshold_default, length(w))
        return (w, gp, ct)
    else
        error("Unsupported validation set entry format: $(typeof(entry))")
    end
end

global set = [_normalize_validation_entry(e) for e in FileIO.load("./ModelBased/validation_set.jld2","set")]
#FileIO.save("./ModelBased/validation_set.jld2","set",set)


global validation_results = FileIO.load("./ModelBased/validation_results.jld2","validation_results")
#FileIO.save("./ModelBased/validation_results.jld2","validation_results",validation_results)


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

    for (w, gp, ct) in set

        reset!(env)
        # Perform validation checks on wind and gp
        global wind = [w]
        global grid_price = gp
        global curtailment_threshold = ct

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

    global set = []

    for i in 1:n
        wind, ct = generate_wind()
        gp = generate_grid_price()
        push!(set, (wind, gp, ct))
    end

    return set
end


function regenerate_validation_grid_prices!(; same_day::Bool = false,
                                             save::Bool = true,
                                             set_path::AbstractString = "./ModelBased/validation_set.jld2")
    global set

    refreshed = Vector{Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}}(undef, length(set))

    for (i, entry) in enumerate(set)
        w, _, ct = _normalize_validation_entry(entry)
        gp = generate_grid_price(; same_day = same_day)

        length(gp) == length(w) || error("Grid price length mismatch at index $i: length(gp)=$(length(gp)) vs length(w)=$(length(w))")
        length(ct) == length(w) || error("Curtailment threshold length mismatch at index $i: length(ct)=$(length(ct)) vs length(w)=$(length(w))")

        refreshed[i] = (Float32.(w), Float32.(gp), Float32.(ct))
    end

    set = refreshed

    if save
        FileIO.save(set_path, "set", set)
    end

    return set
end



