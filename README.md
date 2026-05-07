# Toward an Energy-Optimized Operation of Data Centers Located in Wind Farms Using Reinforcement Learning

This repository contains the public code accompanying the paper **"Toward an Energy-Optimized Operation of Data Centers Located in Wind Farms Using Reinforcement Learning"**.

The code studies reinforcement learning (RL) policies for scheduling data-center compute load around wind availability, curtailment-related free-energy opportunities, and time-varying grid prices.

## Scope

The `Minimal/` experiments are the experiments from the paper.

`ModelBased/` and `Complex/` go beyond the paper:

- `ModelBased/` keeps the same core RL control problem as `Minimal/`, but replaces synthetic signals with model-generated wind/curtailment signals and real grid-price time series.
- `Complex/` moves beyond scalar daily utilization control to discrete job/resource allocation with deadlines, slot constraints, categorical PPO, and a categorical multi-agent transformer policy.

## Repository Structure

- `Project.toml`: top-level Julia environment. The local `RL/` package is resolved through `[sources]`.
- `setup.jl`: project setup script that activates this environment, develops `RL/`, and instantiates dependencies.
- `RL/`: repository-local custom Julia RL package snapshot.
- `Minimal/`: paper experiment setting with synthetic wind and grid-price generators, optimizer baseline, imitation-learning trajectories, validation, and result aggregation.
- `ModelBased/`: extension setting with learned FM/Transformer wind-signal generation, real grid-price loaders, optimizer baseline, validation, and result aggregation.
- `ModelBased/models/`: model runtime, statistics, saved FM model, and grid-price CSV inputs used by the model-based scenario.
- `Complex/`: extension setting for discrete job allocation and multi-agent scheduling.
- `Complex/PPO/`: categorical PPO experiment for the complex allocation setting.
- `Complex/MAT-Categorial/`: categorical multi-agent transformer experiment for the complex allocation setting.
- `*.jld2`: validation sets, test sets, cached results, generated trajectories, or saved agents depending on location.

Active algorithms in `Minimal/` and `ModelBased/` are:

- `SAC`
- `SAC2`
- `PPO`
- `DDPG`

## Algorithms and Training Terms

The paper evaluates actor-critic deep RL methods for continuous control:

- `PPO`: Proximal Policy Optimization. PPO is an on-policy method; it updates from trajectories collected by the current policy.
- `DDPG`: Deep Deterministic Policy Gradient. DDPG is an off-policy deterministic actor-critic method and learns from replay-buffer samples.
- `SAC`: Soft Actor-Critic. SAC is an off-policy stochastic actor-critic method with entropy regularization.
- `SAC2`: in this repository, `SAC2` denotes the SAC variant with the additional on-policy update routine described in the paper. It keeps replay-based SAC updates, but periodically refines the actor and critics on coherent recent rollout windows. In IL mode, the same routine is applied to random contiguous windows from the optimizer-generated expert corpus. This is meant to address the fixed-day credit-assignment problem where early actions are only evaluated through delayed completion feedback.

The paper studies two mitigation strategies for delayed credit assignment:

- `IL`: optimization-based imitation learning. The JuMP/IPOPT optimizer produces expert trajectories. During training, IL updates are not used only as one-time pretraining; they are repeatedly injected into the normal update loop.
- `RS`: potential-based reward shaping. Set `reward_shaping = false` or `reward_shaping = true` in the respective training scripts, or pass `reward_shaping = ...` to `train(...)`, to toggle the potential-based reward shaping used in the paper. The shaping term uses workload progress as potential and redistributes completion-related feedback from the end of the day into intermediate steps.

## Setup

Install Julia, then run from the repository root:

```bash
julia setup.jl
```

The setup script runs:

```julia
using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "RL"))
Pkg.instantiate()
```

For interactive work, start Julia from the repository root with:

```bash
julia --project=.
```

The local RL package is selected by the top-level `Project.toml`:

```toml
[sources]
RL = { path = "RL" }
```

This means the code uses the checked-in `RL/` snapshot instead of a registry package or a user-local development path.

## Minimal Experiments

`Minimal/` is the controlled paper setting. One episode is one day:

- `te = 1440.0` minutes
- `dt = 5.0` minutes
- 288 control steps plus history pre-roll

The environment uses synthetic wind and grid-price generators. The action is continuous utilization in `[-1, 1]`, mapped to `[0, 1]`. Reward combines free wind energy usage, grid-energy cost, and terminal pressure to finish the daily compute workload.

The setting also includes a deterministic JuMP/IPOPT optimizer used as:

- an upper-bound baseline for validation,
- a source of demonstration trajectories for imitation learning.

### Single Training Script

From `julia --project=.`:

```julia
include("Minimal/SAC2/Minimal_SAC2.jl")
train(;
    num_steps = 8_000,
    inner_loops = 1,
    outer_loops = 1,
    optimal_trainings = 0,
    reward_shaping = false,
    plot_runs = false,
)
validate_agent()
save_agent()
```

The main training parameters are:

- `num_steps`: minimum number of environment interaction steps per normal RL training block.
- `inner_loops`: number of normal RL training blocks per outer loop. Each inner loop runs online rollout training for `num_steps`.
- `outer_loops`: number of repeated train/evaluate cycles.
- `optimal_trainings`: number of optimizer-trajectory IL update calls before the `inner_loops` normal RL blocks in each outer loop.
- `reward_shaping`: toggles the potential-based reward shaping analyzed in the paper.

So, with `outer_loops = 10`, `optimal_trainings = 1`, and `inner_loops = 3`, each outer loop first performs one `RL.update_IL(agent.policy, optimal_trajectory)` call and then performs three normal online RL training blocks.

Other available scripts follow the same pattern:

```julia
include("Minimal/SAC/Minimal_SAC.jl")
include("Minimal/PPO/Minimal_PPO.jl")
include("Minimal/DDPG/Minimal_DDPG.jl")
```

To inspect a rollout visually after loading or training an agent:

```julia
render_run()
```

### Batch Runs

The experiment runner handles repeated seeds, reward-shaping variants, IL/no-IL variants, sharded result files, and plotting helpers:

```julia
include("Minimal/SAC2/Minimal_SAC2.jl")
include("Minimal/generate_results.jl")

collect_runs(
    1;
    selected_algorithms = ["SAC2"],
    il_types = ["no_IL"],
    rs_types = ["no_RS"],
)
```

`Minimal/generate_results.jl` uses the following default sweep parameters:

| Algorithm | `inner_loops` | `outer_loops` | `outer_loops_IL` | `optimal_trainings` | `num_steps` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SAC` | 1 | 1000 | 3000 | 160 | 8000 |
| `SAC2` | 3 | 1000 | 3000 | 1 | 8000 |
| `PPO` | 3 | 500 | 5000 | 1 | 8000 |
| `DDPG` | 1 | 1000 | 0 | 160 | 8000 |

`outer_loops_IL` is used instead of `outer_loops` for IL runs. `SAC` and `DDPG` use higher `optimal_trainings` values because their IL updates are off-policy replay-style batch updates; each call consumes fewer coherent expert samples than the on-policy-style SAC2 and PPO updates. SAC2 and PPO therefore use `optimal_trainings = 1` with larger trajectory windows, while SAC and DDPG use `160` smaller off-policy update calls.

Merge sharded result files into `Minimal/training_results.jld2`:

```julia
merge_sharded_results!()
```

Plot or compute validation/test comparisons:

```julia
plot_validation_comparison()
plot_test_comparison()
```

### Optimal Trajectories

Training scripts load `Minimal/optimal_trajectories.jld2` because imitation learning support is wired into the same scripts. If the trajectory cache is missing or stale, regenerate it from the Minimal environment:

```julia
include("Minimal/SAC2/Minimal_SAC2.jl")
include("Minimal/generate_optimal_trajectory.jl")
generate_optimal_trajectories()
```

This can take time because it repeatedly solves daily optimization problems with IPOPT.

## ModelBased Experiments

`ModelBased/` extends the paper setting. It preserves the Minimal control structure while replacing synthetic signals with more realistic inputs:

- wind and curtailment-threshold signals are generated by a local FM/Transformer runtime,
- grid prices are loaded from real seasonal CSV time series,
- generated wind/curtailment sets can be cached and sampled during training,
- validation entries contain wind, grid price, and curtailment-threshold trajectories.

Usage mirrors `Minimal/`:

```julia
include("ModelBased/SAC2/ModelBased_SAC2.jl")
train(;
    num_steps = 8_000,
    inner_loops = 1,
    outer_loops = 1,
    optimal_trainings = 0,
    reward_shaping = false,
    plot_runs = false,
)
validate_agent()
save_agent()
```

Available scripts:

```julia
include("ModelBased/SAC/ModelBased_SAC.jl")
include("ModelBased/SAC2/ModelBased_SAC2.jl")
include("ModelBased/PPO/ModelBased_PPO.jl")
include("ModelBased/DDPG/ModelBased_DDPG.jl")
```

Batch runs:

```julia
include("ModelBased/SAC2/ModelBased_SAC2.jl")
include("ModelBased/generate_results.jl")

collect_runs(
    1;
    selected_algorithms = ["SAC2"],
    il_types = ["no_IL"],
    rs_types = ["no_RS"],
)
```

`ModelBased/generate_results.jl` uses the following default sweep parameters:

| Algorithm | `inner_loops` | `outer_loops` | `outer_loops_IL` | `optimal_trainings` | `num_steps` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SAC` | 1 | 1000 | 2000 | 160 | 8000 |
| `SAC2` | 3 | 1000 | 3000 | 1 | 8000 |
| `PPO` | 3 | 500 | 5000 | 1 | 8000 |
| `DDPG` | 1 | 1000 | 2000 | 160 | 8000 |

The interpretation is the same as in `Minimal/`: `optimal_trainings` controls how many IL update calls are inserted before the normal `inner_loops`, and `outer_loops_IL` is the outer-loop count used for IL configurations.

Regenerate optimizer trajectories when needed:

```julia
include("ModelBased/SAC2/ModelBased_SAC2.jl")
include("ModelBased/generate_optimal_trajectory.jl")
generate_optimal_trajectories()
```

## Complex Experiments

`Complex/` is an extension beyond the paper. It changes the problem from daily scalar load scheduling to discrete operational allocation:

- multiple job slots,
- stochastic job arrivals,
- per-job remaining load,
- deadlines and missed-deadline penalties,
- integer allocation slots,
- categorical action decoding,
- optional cooperative multi-agent state/action structure.

Run categorical PPO:

```julia
include("Complex/PPO/Complex_PPO.jl")
train(; num_steps = 50_000)
save_agent()
```

Run categorical multi-agent transformer training:

```julia
include("Complex/MAT-Categorial/Complex_MAT_Categorial.jl")
train(; num_steps = 50_000)
save_agent()
```

The default `train` value in `Complex/` is intentionally large. Use smaller `num_steps` values for smoke tests.

## Saved Artifacts and Generated Files

Most long-running outputs are local artifacts and are ignored by git:

- `training_results.jld2`
- `training_results_*.jld2`
- `optimal_trajectories.jld2`
- `saves/`
- `training_frames/`
- videos and old result folders

Validation and test data included in the experiment folders are used for reproducible comparisons. Do not overwrite them unless you intentionally want to regenerate the evaluation set.

## Reproducibility Notes

- Start Julia from the repository root with `julia --project=.`.
- Use `setup.jl` after a fresh checkout or dependency change.
- Training scripts set their own random seeds unless you edit the script-local `seed` assignment.
- Full training runs can be long. For smoke tests, reduce `num_steps`, `outer_loops`, and selected algorithms.
- `ModelBased/` depends on the model runtime and data files under `ModelBased/models/`.
- `Minimal/` is the paper reproduction setting. `ModelBased/` and `Complex/` are included as research extensions beyond the paper.
