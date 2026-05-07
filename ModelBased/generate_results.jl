using JLD2
using FileIO
using PlotlyJS
using Statistics

# File path for saving results
results_file = joinpath(@__DIR__, "training_results.jld2")
results_shard_prefix = "training_results__run_"

new_results_dict() = Dict{String, Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}}()

function load_or_init_results(file_path::String)
    if isfile(file_path)
        loaded = FileIO.load(file_path, "results")
        println("Loaded existing results file: $file_path")
        return loaded
    end
    created = new_results_dict()
    FileIO.save(file_path, "results", created)
    println("Created new results dictionary: $file_path")
    return created
end

function make_unique_results_file(; prefix::String = results_shard_prefix)
    ts = floor(Int, time())
    pid = getpid()
    counter = 0
    while true
        suffix = counter == 0 ? "" : "_$(counter)"
        candidate = joinpath(@__DIR__, "$(prefix)$(ts)_pid$(pid)$(suffix).jld2")
        if !isfile(candidate)
            return candidate
        end
        counter += 1
    end
end

function find_results_shards(; prefix::String = results_shard_prefix, include_main::Bool = false)
    shards = sort(filter(path -> startswith(basename(path), prefix) && endswith(path, ".jld2"),
                         readdir(@__DIR__; join = true)))
    if include_main
        return shards
    end
    return filter(path -> abspath(path) != abspath(results_file), shards)
end

function merge_results_into!(target_results, source_results)
    merged_runs = 0
    collision_count = 0

    for alg_name in keys(source_results)
        if alg_name == "Optimal"
            # Keep existing target Optimal cache unless absent.
            if !haskey(target_results, "Optimal")
                target_results["Optimal"] = deepcopy(source_results["Optimal"])
            end
            continue
        end
        alg_name in ACTIVE_ALGORITHM_NAMES || continue

        for il_type in keys(source_results[alg_name])
            for rs_type in keys(source_results[alg_name][il_type])
                ensure_nested_dict!(target_results, alg_name, il_type, rs_type)
                for seed in sort(collect(keys(source_results[alg_name][il_type][rs_type])))
                    target_seed = seed
                    while haskey(target_results[alg_name][il_type][rs_type], target_seed)
                        target_seed += 1
                        collision_count += 1
                    end
                    target_results[alg_name][il_type][rs_type][target_seed] =
                        deepcopy(source_results[alg_name][il_type][rs_type][seed])
                    merged_runs += 1
                end
            end
        end
    end

    return merged_runs, collision_count
end

function merge_sharded_results!(;
    target_file::String = results_file,
    shard_files::Vector{String} = find_results_shards(),
    delete_shards_after_merge::Bool = false
)
    target_results = load_or_init_results(target_file)

    if isempty(shard_files)
        println("No shard result files found for merge.")
        return target_results
    end

    total_merged = 0
    total_collisions = 0

    for shard_file in shard_files
        if abspath(shard_file) == abspath(target_file)
            continue
        end
        if !isfile(shard_file)
            println("Skipping missing shard file: $shard_file")
            continue
        end

        shard_results = FileIO.load(shard_file, "results")
        merged, collisions = merge_results_into!(target_results, shard_results)
        total_merged += merged
        total_collisions += collisions

        if delete_shards_after_merge
            rm(shard_file; force = true)
        end
        println("Merged $(merged) runs from $(basename(shard_file))")
    end

    FileIO.save(target_file, "results", target_results)
    global results = target_results

    println("Merge complete. Added $total_merged runs into $target_file.")
    if total_collisions > 0
        println("Seed collisions handled by incrementing seed ids: $total_collisions")
    end

    return target_results
end

# Initialize or load results dictionary with hierarchical structure
if isfile(results_file)
    results = FileIO.load(results_file, "results")
    println("Loaded existing results file")
else
    # Now we have another level in the hierarchy for reward_shaping
    results = new_results_dict()
    println("Created new results dictionary")
    FileIO.save(results_file, "results", results)
end

# List of algorithms to test
algorithms = [
    ("SAC", "ModelBased/SAC/ModelBased_SAC.jl"),
    ("SAC2", "ModelBased/SAC2/ModelBased_SAC2.jl"),
    ("PPO", "ModelBased/PPO/ModelBased_PPO.jl"),
    ("DDPG", "ModelBased/DDPG/ModelBased_DDPG.jl")
]
const ACTIVE_ALGORITHM_NAMES = first.(algorithms)

function prune_inactive_results!(results)
    changed = false
    for alg_name in collect(keys(results))
        if alg_name == "Optimal" || alg_name == "Untrained" || alg_name in ACTIVE_ALGORITHM_NAMES
            continue
        end
        delete!(results, alg_name)
        changed = true
    end
    return changed
end

if prune_inactive_results!(results)
    FileIO.save(results_file, "results", results)
end

# algorithms = [
#     ("SAC", "ModelBased/SAC/ModelBased_SAC.jl"),
#     ("SAC2", "ModelBased/SAC2/ModelBased_SAC2.jl"),
#     ("PPO", "ModelBased/PPO/ModelBased_PPO.jl"),
#     ("DDPG", "ModelBased/DDPG/ModelBased_DDPG.jl")
# ]

# Function to ensure nested dictionary structure exists
function ensure_nested_dict!(results, alg_name, il_type, reward_shaping)
    if !haskey(results, alg_name)
        results[alg_name] = Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}()
    end
    if !haskey(results[alg_name], il_type)
        results[alg_name][il_type] = Dict{String, Dict{Int, Dict{String, Any}}}()
    end
    if !haskey(results[alg_name][il_type], reward_shaping)
        results[alg_name][il_type][reward_shaping] = Dict{Int, Dict{String, Any}}()
    end
end

trajectories_file = joinpath(@__DIR__, "optimal_trajectories.jld2")
trajectories = FileIO.load(trajectories_file, "trajectories")

function collect_runs(
    n = 5;
    selected_algorithms::Vector{String} = String[],
    il_types::Vector{String} = ["no_IL", "IL"],
    rs_types::Vector{String} = ["with_RS", "no_RS"],
    run_results_file::Union{Nothing, String} = nothing,
    use_unique_results_file::Bool = true
)
    target_results_file = if isnothing(run_results_file)
        use_unique_results_file ? make_unique_results_file() : results_file
    else
        abspath(run_results_file)
    end
    run_results = load_or_init_results(target_results_file)
    println("collect_runs will write to: $target_results_file")

    # Filter algorithms based on input or use all if none specified
    algs_to_run = if isempty(selected_algorithms)
        algorithms
    else
        filter(a -> a[1] in selected_algorithms, algorithms)
    end
    
    # Run training for each algorithm
    for (alg_name, script_path) in algs_to_run
        println("\n=== Testing $alg_name ===")
        
        # Run all requested combinations of IL and reward_shaping
        for il_type in il_types
            for rs_type in rs_types
                println("\nRunning $(il_type) with $(rs_type):")
                ensure_nested_dict!(run_results, alg_name, il_type, rs_type)
                
                for i in 1:n
                    println("\nStarting training run $i")
                    global seed = i
                    include(script_path)

                    agent_save = nothing
                    
                    # Algorithm-specific default parameters
                    default_params = Dict(
                        "SAC" => Dict(
                            "inner_loops" => 1,
                            "outer_loops" => 1000,
                            "outer_loops_IL" => 2000,
                            "optimal_trainings" => 160,
                            "num_steps" => 8_000
                        ),
                        "SAC2" => Dict(
                            "inner_loops" => 3,
                            "outer_loops" => 1000,
                            "outer_loops_IL" => 3000,
                            "optimal_trainings" => 1,
                            "num_steps" => 8_000
                        ),
                        "PPO" => Dict(
                            "inner_loops" => 3,
                            "outer_loops" => 500,
                            "outer_loops_IL" => 5000,
                            "optimal_trainings" => 1,
                            "num_steps" => 8_000
                        ),
                        "DDPG" => Dict(
                            "inner_loops" => 1,
                            "outer_loops" => 1000,
                            "outer_loops_IL" => 2000,
                            "optimal_trainings" => 160,
                            "num_steps" => 8_000
                        )
                    )

                    outer_loops_this_run = il_type == "IL" ?
                        default_params[alg_name]["outer_loops_IL"] :
                        default_params[alg_name]["outer_loops"]

                    # Set appropriate training parameters
                    train_params = Dict{Symbol,Any}(
                        :inner_loops => default_params[alg_name]["inner_loops"],
                        :outer_loops => outer_loops_this_run,
                        :num_steps => default_params[alg_name]["num_steps"],
                        :plot_runs => false,
                        :reward_shaping => (rs_type == "with_RS")
                    )

                    if il_type == "IL"
                        train_params[:optimal_trainings] = default_params[alg_name]["optimal_trainings"]
                        global optimal_trajectory = trajectories[alg_name][rs_type]
                    else
                        train_params[:optimal_trainings] = 0
                    end
                    
                    # Run training with parameters
                    train(;train_params...)
                    
                    # Store only the policies from the agents
                    agent_policy = deepcopy(agent.policy)
                    agent_save_policy = isnothing(agent_save) ? nothing : deepcopy(agent_save.policy)
                    
                    # Store results
                    run_results[alg_name][il_type][rs_type][seed] = Dict(
                        "agent_save_policy" => agent_save_policy,
                        "agent_policy" => agent_policy,
                        "rewards" => hook.rewards,
                        "validation_scores" => validation_scores
                    )
                    
                    FileIO.save(target_results_file, "results", run_results)
                    # Keep only serialized/copied results; release run-local heavy objects.
                    agent_policy = nothing
                    agent_save_policy = nothing
                    train_params = nothing
                    default_params = nothing
                    global agent = nothing
                    global agent_save = nothing
                    global hook = nothing
                    GC.gc(true)
                    println("Saved results for $alg_name ($il_type, $rs_type) with seed $seed")
                end
            end
        end
    end

    println("\nAll training runs completed. Results saved to $target_results_file")

    # Print structure summary
    println("\nFinal results structure:")
    for alg in keys(run_results)
        println("Algorithm: $alg")
        for il_type in keys(run_results[alg])
            for rs_type in keys(run_results[alg][il_type])
                n_seeds = length(keys(run_results[alg][il_type][rs_type]))
                println("  └─ $il_type / $rs_type: $n_seeds seeds")
            end
        end
    end

    return target_results_file
end





function clean_reconstructed_policies!()
    println("Starting policy type check...")
    deleted_count = 0
    
    # Go through all algorithms
    for alg_name in keys(results)
        # Skip the Optimal results as they don't have policies
        alg_name == "Optimal" && continue
        alg_name in ACTIVE_ALGORITHM_NAMES || continue
        
        # Expected policy type for each algorithm
        expected_type = if alg_name == "PPO"
            PPOPolicy
        elseif alg_name == "SAC"
            SACPolicy
        elseif alg_name == "DDPG"
            CustomDDPGPolicy
        else
            continue  # Skip unknown algorithms
        end
        
        # Go through all IL variants
        for il_type in keys(results[alg_name])
            # Go through all reward shaping variants
            for rs_type in keys(results[alg_name][il_type])
                seeds_to_delete = Int[]
                
                # Check each seed
                for seed in keys(results[alg_name][il_type][rs_type])
                    policy = results[alg_name][il_type][rs_type][seed]["agent_policy"]
                    
                    # Check if policy is reconstructed
                    if typeof(policy) != expected_type && contains(string(typeof(policy)), "ReconstructedMutable")
                        push!(seeds_to_delete, seed)
                        deleted_count += 1
                        println("Found reconstructed policy in $alg_name ($il_type, $rs_type) seed $seed")
                    end
                end
                
                # Delete identified seeds
                for seed in seeds_to_delete
                    delete!(results[alg_name][il_type][rs_type], seed)
                end
            end
        end
    end
    
    # Save the cleaned results
    FileIO.save(results_file, "results", results)
    println("\nCleaning complete: Removed $deleted_count reconstructed policies")
    println("Updated results saved to $results_file")
end


function plot_validation_comparison(; current = false)
    # Include validation script
    include("ModelBased/Validation_ModelBased.jl")
    
    # Ensure results dictionary is loaded
    if !@isdefined(results)
        if isfile(results_file)
            global results = FileIO.load(results_file, "results")
            println("Loaded existing results file")
        else
            global results = Dict{String, Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}}()
            println("Created new results dictionary")
        end
    end
    
    # Check for cached optimal baseline scores from various sources
    optimal_scores = if haskey(results, "Optimal") && 
                       haskey(results["Optimal"], "baseline") && 
                       haskey(results["Optimal"]["baseline"], "scores") &&
                       haskey(results["Optimal"]["baseline"]["scores"], 1)
        println("Using cached optimal baseline scores from results...")
        results["Optimal"]["baseline"]["scores"][1]["data"]
    elseif @isdefined(validation_results) && haskey(validation_results, "optimizer")
        println("Using optimal baseline scores from validation_results...")
        validation_results["optimizer"]
    else
        println("Computing optimal baseline scores...")
        global optimal_scores = validate_agent(optimizer = true)
        # Cache the optimal scores with proper nested structure
        if !haskey(results, "Optimal")
            results["Optimal"] = Dict{String, Dict{String, Dict{String, Dict{Int, Dict{String, Any}}}}}()
        end
        if !haskey(results["Optimal"], "baseline")
            results["Optimal"]["baseline"] = Dict{String, Dict{Int, Dict{String, Any}}}()
        end
        if !haskey(results["Optimal"]["baseline"], "scores")
            results["Optimal"]["baseline"]["scores"] = Dict{Int, Dict{String, Any}}()
        end
        results["Optimal"]["baseline"]["scores"][1] = Dict{String, Any}("data" => optimal_scores)
        # Save updated results to file
        FileIO.save(results_file, "results", results)
        println("Saved optimal baseline scores to results file")
        optimal_scores
    end
    
    # Initialize dictionary to store all validation results
    global best_validation_results = Dict{String, Any}()
    best_validation_results["Optimal"] = (optimal_scores, nothing, nothing)
    
    # Add untrained baseline if available
    if @isdefined(validation_results) && haskey(validation_results, "untrained")
        println("Adding untrained baseline scores...")
        best_validation_results["Untrained"] = (validation_results["untrained"], nothing, nothing)
    end
    
    # Go through all results and validate each agent
    for alg_name in keys(results)
        # Skip the Optimal results as they're handled separately
        alg_name == "Optimal" && continue
        alg_name in ACTIVE_ALGORITHM_NAMES || continue
        
        for il_type in keys(results[alg_name])
            for rs_type in keys(results[alg_name][il_type])
                # Collect scores for all seeds of this configuration
                config_scores = Dict{Int, Vector{Float32}}()
                config_means = Dict{Int, Float64}()
                config_timelines = Dict{Int, Vector{Float32}}()
                
                for seed in keys(results[alg_name][il_type][rs_type])
                    # Get the saved policy
                    saved_policy = results[alg_name][il_type][rs_type][seed]["agent_save_policy"]
                    
                    # Skip if policy is nothing
                    if isnothing(saved_policy)
                        println("Skipping $(alg_name)-$(il_type)-$(rs_type)-seed$(seed) (no policy saved)")
                        continue
                    end
                    
                    # Construct new agent with saved policy
                    global agent = Agent(saved_policy, Trajectory())
                    
                    # Run validation
                    println("Validating $(alg_name)-$(il_type)-$(rs_type)-seed$(seed)...")
                    scores = validate_agent()
                    
                    # Store scores, mean and timeline
                    config_scores[seed] = scores
                    config_means[seed] = mean(scores)
                    config_timelines[seed] = results[alg_name][il_type][rs_type][seed]["validation_scores"]
                end
                
                # Find the best seed based on mean score
                if !isempty(config_means)
                    best_mean = maximum(values(config_means))
                    best_seeds = sort([
                        seed for (seed, seed_mean) in config_means
                        if isapprox(seed_mean, best_mean; atol = 1e-10, rtol = 1e-8)
                    ])
                    if isempty(best_seeds)
                        best_seeds = [argmax(config_means)]
                    end
                    best_seed = first(best_seeds)
                    key = "$(alg_name)-$(il_type)-$(rs_type)"
                    best_validation_results[key] = (
                        config_scores[best_seed],
                        config_timelines[best_seed],
                        best_seeds
                    )
                end
            end
        end
    end

    global agent_save
    if current 
        if !isnothing(agent_save)
            global agent, validation_scores
            agent_temp = deepcopy(agent)
            agent = deepcopy(agent_save)

            current_scores = validate_agent()

            best_validation_results["Current"] = (
                            current_scores,
                            validation_scores,
                            nothing
                        )
            
            agent = deepcopy(agent_temp)
        else
            println("Current failed: agent_save is nothing!")
        end
    end
    
    # First, calculate the order based on mean scores
    order = sort(collect(keys(best_validation_results)), 
                by=key->mean(best_validation_results[key][1]), 
                rev=true)  # descending order

    # Create color mapping for consistent colors across plots
    color_map = Dict{String, Vector{Int}}()
    
    # Define colors for special cases
    color_map["Optimal"] = [76, 175, 80]  # Muted forest green
    color_map["Untrained"] = [239, 83, 80]  # Reddish color
    color_map["Current"] = [239, 83, 239]  # Purple
    
    # Define algorithm colors
    for key in order
        if key != "Optimal" && key != "Untrained"
            parts = split(key, "-")
            alg = parts[1]
            il_type = parts[2]
            rs_type = parts[3]
            
            # Define semi-muted, aesthetic base colors for algorithms
            base_color = if alg == "SAC"
                [184, 71, 82]    # Richer burgundy
            elseif alg == "PPO"
                [98, 150, 209]   # Brighter steel blue
            else  # DDPG
                [168, 119, 175]  # Brighter purple
            end
            
            # IL affects hue
            if il_type == "IL"
                base_color = round.(Int, base_color .* 1.3)
            end
            
            # RS affects saturation
            if rs_type == "with_RS"
                luminance = sum(base_color) / 3
                base_color = round.(Int, base_color .* 0.5 .+ luminance * 0.5)
            end
            
            color_map[key] = base_color
        end
    end

    # Create first plot (validation scores)
    traces1 = AbstractTrace[]

    format_seed_label(key, seeds) = begin
        if isnothing(seeds) || isempty(seeds)
            key
        elseif length(seeds) == 1
            "$(key) (seed=$(seeds[1]))"
        else
            "$(key) (seeds=$(join(seeds, ", ")))"
        end
    end
    
    # Add traces in order
    for key in order
        scores, _, best_seeds = best_validation_results[key]
        display_name = format_seed_label(key, best_seeds)
        color = color_map[key]
        
        push!(traces1, box(
            y=scores,
            name=display_name,
            boxpoints="all",
            quartilemethod="linear",
            marker_color="rgb($(color[1]), $(color[2]), $(color[3]))",
            boxmean=true
        ))
    end
    
    # Create and display the validation scores plot
    layout1 = Layout(
        title="Algorithm Performance Comparison",
        yaxis_title="Validation Score",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)
    )
    
    p1 = plot(traces1, layout1)
    display(p1)

    # Create second plot (validation timelines)
    traces2 = AbstractTrace[]
    
    # Add traces in the same order
    for key in order
        _, timeline, best_seeds = best_validation_results[key]
        
        # Skip if timeline is nothing
        if !isnothing(timeline)
            color = color_map[key]
            display_name = format_seed_label(key, best_seeds)
            
            push!(traces2, scatter(
                y=timeline,
                name=display_name,
                mode="lines",
                line_color="rgb($(color[1]), $(color[2]), $(color[3]))"
            ))
        end
    end
    
    # Create and display the timeline plot
    layout2 = Layout(
        title="Validation Score Timeline During Training",
        yaxis_title="Validation Score",
        xaxis_title="Training Steps",
        showlegend=true,
        legend=attr(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=attr(b=100)
    )
    
    p2 = plot(traces2, layout2)
    display(p2)
    
    #return p1, p2  # Return both plot objects
end
