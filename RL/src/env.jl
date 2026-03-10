mutable struct GeneralEnv <: AbstractEnv
    te
    t0
    dt

    f
    do_step
    featurize
    prepare_action
    reward_function

    state_space
    action_space
    sim_space
    
    y0
    y
    state

    action0
    action
    delta_action
    p

    steps
    time

    reward
 
    terminated
    truncated
    terminated_at_timeout


    max_value
    check_max_value
end

function GeneralEnv(; te = 2.0, 
                  t0 = 0.0, 
                  dt = 0.005, 
                  f = nothing,
                  do_step = nothing,
                  featurize = nothing, 
                  prepare_action = nothing, 
                  reward_function = nothing, 
                  state_space = nothing, 
                  action_space, 
                  sim_space, 
                  y0 = nothing, 
                  action0 = nothing, 
                  reward = 0.0, 
                  max_value = 20.0,
                  check_max_value = "y",
                  use_gpu = false,
                  terminated_at_timeout = false,)

    if isnothing(f)
        f = function(u = nothing, p = nothing, t = nothing; env = nothing)
            return 0.0
        end
    end
    
    if isnothing(featurize)
        featurize = function(y0 = nothing, t0 = nothing; env = nothing) 
            if isnothing(env)
                return use_gpu ? CuArray(y0) : y0
            else
                return env.y
            end
        end
    end

    if isnothing(prepare_action)
        prepare_action = function(action0 = nothing, t0 = nothing; env = nothing) 
            if isnothing(env)
                return use_gpu ? CuArray(action0) : action0
            else
                return use_gpu ? CuArray(env.action) : env.action
            end
        end
    end

    if isnothing(reward_function)
        reward_function = function(env) 
            return env.reward
        end
    end

    if isnothing(y0)
        y0 = use_gpu ? CuArray(zeros(size(sim_space))) : zeros(size(sim_space))
    end

    y = y0
    state = featurize(y0,t0)
    
    if isnothing(state_space)
        state_space = Space(fill(-1..1, size(state)))
    end

    if isnothing(action0)
        action0 = zeros(size(action_space))
    end

    action = action0
    delta_action = zeros(size(action_space))
    p = prepare_action(action0, t0)

    steps = 0
    time = 0.0

    if isnothing(reward)
        reward = 0.0
    end
 
    terminated = false
    truncated = false

    GeneralEnv(te, 
           t0, 
           dt, 
           f,
           do_step,
           featurize, 
           prepare_action, 
           reward_function, 
           state_space, 
           action_space, 
           sim_space, 
           y0, 
           y, 
           state, 
           action0, 
           action, 
           delta_action,
           p, 
           steps, 
           time, 
           reward, 
           terminated,
           truncated,
           terminated_at_timeout,
           max_value,
           check_max_value)
end


action_space(env::GeneralEnv) = env.action_space
state_space(env::GeneralEnv) = env.state_space

reward(env::GeneralEnv) = env.reward
is_terminated(env::GeneralEnv) = env.terminated
is_truncated(env::GeneralEnv) = env.truncated
state(env::GeneralEnv) = env.state

function reset!(env::GeneralEnv)
    env.y = env.y0
    env.state = env.featurize(env.y0, env.t0)
    env.action = env.action0
    env.delta_action = zeros(size(env.action0))
    env.p = env.prepare_action(env.action0, env.t0)
    env.steps = 0
    env.time = 0.0
    env.reward = 0
    env.terminated = false
    env.truncated = false
    nothing
end

function (env::GeneralEnv)(action; reward_shaping = nothing)
    env.delta_action = action - env.action
    env.action = action
    
    env.p = env.prepare_action(; env = env)

    if isnothing(reward_shaping)
        env.y = env.do_step(env)
    else
        env.y = env.do_step(env; reward_shaping = reward_shaping)
    end

    # Legacy code - use this inside of do_step in script if needed

    # if isnothing(env.do_step)
    #     if env.use_radau
    #         tspan = (env.time, env.time + env.dt)
    #         prob = ODEProblem(env.f, env.y, tspan, env.p)
    #         sol = solve(prob, RadauIIA5(), reltol=1e-8, abstol=1e-8)
    #         env.y = last(sol.u)
    #     else
    #         dt_temp = env.dt / env.oversampling
            
    #         for i in 1:env.oversampling
    #             y_old = env.y
    #             env.y = env.y + 0.5 * dt_temp * env.f(;env=env)
    #             env.y = y_old + dt_temp * env.f(;env=env)
    #         end
    #     end
    # else
    #     env.y = env.do_step(env)
    # end

    env.reward = env.reward_function(env)

    env.state = env.featurize(; env = env)

    env.steps += 1
    env.time += env.dt

    if !env.terminated
        if env.check_max_value == "y"
            env.terminated = maximum(abs.(env.y)) > env.max_value
            if maximum(abs.(env.y)) > env.max_value
                println("terminated early at $(env.steps) steps")
            end
        elseif env.check_max_value == "reward"
            env.terminated = maximum(abs.(env.reward)) > env.max_value
            if maximum(abs.(env.reward)) > env.max_value
                println("terminated early at $(env.steps) steps")
            end
        end
    end

    if env.time >= env.te
        if env.terminated_at_timeout
            env.terminated = true
        else
            env.truncated = true
        end
    end
end