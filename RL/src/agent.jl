"""
    Agent(;kwargs...)

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between an agent and an environment
"""
Base.@kwdef struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory} <: AbstractPolicy
    policy::P
    trajectory::T
end


(agent::Agent)(env) = agent.policy(env)

#####
# Default behaviors
#####

"""
Here we extend the definition of `(p::AbstractPolicy)(::AbstractEnv)` in
`RLBase` to accept an `AbstractStage` as the first argument. Algorithm designers
may customize these behaviors respectively by implementing:

- `(p::YourPolicy)(::AbstractStage, ::AbstractEnv)`
- `(p::YourPolicy)(::PreActStage, ::AbstractEnv, action)`

The default behaviors for `Agent` are:

1. Update the inner `trajectory` given the context of `policy`, `env`, and
   `stage`.
  1. By default we do nothing.
  2. In `PreActStage`, we `push!` the current **state** and the **action** into
     the `trajectory`.
  3. In `PostActStage`, we query the `reward` and `is_terminated` info from
     `env` and push them into `trajectory`.
  4. In the `PosEpisodeStage`, we push the `state` at the end of an episode and
     a dummy action into the `trajectory`.
  5. In the `PreEpisodeStage`, we pop out the latest `state` and `action` pair
     (which are dummy ones) from `trajectory`.

2. Update the inner `policy` given the context of `trajectory`, `env`, and
   `stage`.
  1. By default, we only `update!` the `policy` in the `PreActStage`. And it's
     dispatched to `update!(policy, trajectory, env, stage)`.
"""
function (agent::Agent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.policy, agent.trajectory, env, stage)
end

function (agent::Agent)(stage::PreExperimentStage, env::AbstractEnv)
    update!(agent.policy, agent.trajectory, env, stage)
end

function (agent::Agent)(stage::PreActStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    update!(agent.policy, agent.trajectory, env, stage)
end

function update!(
    ::AbstractPolicy,
    ::AbstractTrajectory,
    ::AbstractEnv,
    ::AbstractStage,
) end

#####
# Default behaviors for known trajectories
#####

function update!(
    ::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::AbstractStage,
) end

function update!(
    trajectory::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
end

function update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = state(env)

    push!(trajectory[:state], s)
    push!(trajectory[:action], action)
end

function update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = reward(env)

    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
end

function update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
end