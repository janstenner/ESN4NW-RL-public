module RL


# imports

using CUDA
using LinearAlgebra
using IntervalSets
using DataFrames
using Statistics
using Distributions
using StructArrays
using UnicodePlots:lineplot
using Flux
using Optimisers
using Adapt
using Random
using CircularArrayBuffers
using ElasticArrays
using MacroTools: @forward
using MacroTools
using ProgressMeter
using StableRNGs
using Setfield: @set
using Zygote
using ChainRulesCore
using Static
using CoherentNoise
using StatsBase




# abstract types

abstract type AbstractEnv end
abstract type AbstractPolicy end
abstract type AbstractHook end
abstract type AbstractStage end
abstract type AbstractTrajectory end




# includes

include("./stages.jl")
include("./device.jl")
include("./base.jl")
include("./env.jl")
include("./hook.jl")
include("./trajectory.jl")
include("./agent.jl")
include("./stop_conditions.jl")
include("./run.jl")
include("./custom_nna.jl")
include("./agent_ddpg.jl")
include("./multi_thread_env.jl")
include("./agent_ppo.jl")
include("./agent_ppo2.jl")
include("./agent_ppo3.jl")
include("./agent_flowppo.jl")
include("./agent_sac.jl")
include("./agent_sac2.jl")
include("./agent_mat.jl")
include("./agent_mat_categorial.jl")
#include("./agent_matmix.jl")



# code to export all, taken from https://discourse.julialang.org/t/exportall/4970/18
for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end
# module
