using FileIO, JLD2
using Dates
using DataFrames
using JSON
using Random
using Flux, NNlib, LinearAlgebra, Statistics

const SERIAL_DEFAULT = "1011089"
const MODELS_DIR = @__DIR__
const SAVE_DIR_FM = joinpath(MODELS_DIR, "saves_HK_FM")
const STATS_DIR = joinpath(MODELS_DIR, "HK_blocks")
const STATS_LOG1P_PATH = joinpath(STATS_DIR, "stats_log1p_cache.json")

const ORDERED_BASE = [
    "wind_mean_ms", "wind_max_ms", "wind_min_ms",
    "rpm_mean", "rpm_max", "rpm_min",
    "power_mean_kw", "power_max_kw", "power_min_kw",
    "power_avail_wind_mean_kw",
    "power_avail_tech_mean_kw",
    "power_avail_force_maj_mean_kw",
    "power_avail_ext_mean_kw",
]
const SEASON_NAMES = ("Winter", "Fruehling", "Sommer", "Herbst")
const READABLE_FLOAT_FEATS = Set([
    "wind_mean_ms", "wind_max_ms", "wind_min_ms",
    "rpm_mean", "rpm_max", "rpm_min",
])

const D_IN = 19
const D_OUT = 13
const NUM_IDX = 7:(6 + D_OUT)
const D_MODEL = 66
const N_HEAD = 6
const N_LAY = 5
const D_FF = 128
const DROPOUT = 0.1
const RANK = 4
const CTX = 5

include(joinpath(@__DIR__, "HK_model.jl"))

const LOG1P_CACHE = Dict{String, NamedTuple{(:scale, :mu_z, :sig_z), Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}}}()

@inline function season_onehot(s::AbstractString)::NTuple{4, Float32}
    return (
        Float32(s == "Winter"),
        Float32(s == "Fruehling"),
        Float32(s == "Sommer"),
        Float32(s == "Herbst"),
    )
end

@inline function floor_slot_index(t::Time)::Int
    minutes_of_day = hour(t) * 60 + minute(t)
    return fld(minutes_of_day, 10) + 1
end

@inline function nearest_slot_from_sc(s::Real, c::Real)::Int
    bestk = 1
    bestd = typemax(Float32)
    @inbounds for k in 1:SLOTS
        ds = Float32(TIME_LUT[1, k]) - Float32(s)
        dc = Float32(TIME_LUT[2, k]) - Float32(c)
        d = ds * ds + dc * dc
        if d < bestd
            bestd = d
            bestk = k
        end
    end
    return bestk
end

function load_log1p_norm(; serial::AbstractString = SERIAL_DEFAULT, stats_log1p_path::AbstractString = STATS_LOG1P_PATH)
    key = String(serial)
    if haskey(LOG1P_CACHE, key)
        return LOG1P_CACHE[key]
    end

    raw = JSON.parsefile(stats_log1p_path)
    haskey(raw, key) || error("Serial $key not found in $stats_log1p_path")
    per_serial = raw[key]

    scale = Vector{Float32}(undef, D_OUT)
    mu_z = Vector{Float32}(undef, D_OUT)
    sig_z = Vector{Float32}(undef, D_OUT)

    @inbounds for (i, feat) in enumerate(ORDERED_BASE)
        haskey(per_serial, feat) || error("Missing feature $feat in $stats_log1p_path for serial $key")
        feat_obj = per_serial[feat]
        scale[i] = Float32(feat_obj["scale"])
        mu_z[i] = Float32(feat_obj["mu_z"])
        sig_z[i] = max(Float32(feat_obj["sig_z"]), 1f-6)
    end

    ln = (scale = scale, mu_z = mu_z, sig_z = sig_z)
    LOG1P_CACHE[key] = ln
    return ln
end

function inverse_log1p_zscore(pred_z::AbstractMatrix{<:Real}; serial::AbstractString = SERIAL_DEFAULT, stats_log1p_path::AbstractString = STATS_LOG1P_PATH)
    size(pred_z, 1) == D_OUT || error("Expected first dimension = D_OUT = $D_OUT, got $(size(pred_z, 1))")
    ln = load_log1p_norm(; serial = serial, stats_log1p_path = stats_log1p_path)
    out = Matrix{Float32}(undef, D_OUT, size(pred_z, 2))

    @inbounds for j in 1:D_OUT
        s = ln.scale[j]
        μ = ln.mu_z[j]
        σ = ln.sig_z[j]
        @views out[j, :] .= s .* expm1.(Float32.(pred_z[j, :]) .* σ .+ μ)
        @views out[j, :] .= max.(out[j, :], 0f0)
    end
    return out
end

function load_fm_model(; serial::AbstractString = SERIAL_DEFAULT, model_dir::AbstractString = SAVE_DIR_FM, assign_globals::Bool = true)
    model_path = joinpath(model_dir, string(serial, ".jld2"))
    isfile(model_path) || error("Model file not found: $model_path")
    data = FileIO.load(model_path)
    loaded_model = data["model"]
    loaded_model isa FMTransformer || error("Loaded model is not FMTransformer. Got: $(typeof(loaded_model))")

    if assign_globals
        global model = loaded_model
        global losses = get(data, "losses", Float32[])
    end
    return loaded_model
end

function build_seed_context(season::AbstractString = "Winter";
                            ctx::Int = CTX,
                            day_start::Time = Time(0, 0),
                            numeric_seed::Union{Nothing, AbstractVector{<:Real}} = nothing)
    season in SEASON_NAMES || error("Unknown season=$season. Use one of $(collect(SEASON_NAMES)).")

    slot_start = floor_slot_index(day_start)
    last_slot = slot_start == 1 ? SLOTS : (slot_start - 1)
    slots = Vector{Int}(undef, ctx)
    @inbounds for i in 1:ctx
        slots[i] = mod1(last_slot - (ctx - i), SLOTS)
    end

    seed_numeric = isnothing(numeric_seed) ? zeros(Float32, D_OUT) : Float32.(numeric_seed)
    length(seed_numeric) == D_OUT || error("numeric_seed must have length D_OUT=$D_OUT")

    oh = season_onehot(season)
    ctx_mat = Matrix{Float32}(undef, D_IN, ctx)

    @inbounds for col in 1:ctx
        ctx_mat[1:4, col] .= (oh[1], oh[2], oh[3], oh[4])
        slot = slots[col]
        ctx_mat[5, col] = TIME_LUT[1, slot]
        ctx_mat[6, col] = TIME_LUT[2, slot]
        ctx_mat[NUM_IDX, col] .= seed_numeric
    end

    return ctx_mat
end

function generate_slots_z(model::FMTransformer;
                          n_slots::Int = SLOTS,
                          season::AbstractString = "Winter",
                          ctx::Int = CTX,
                          day_start::Time = Time(0, 0),
                          steps::Int = 38,
                          seed::Union{Nothing, Int} = nothing,
                          numeric_seed::Union{Nothing, AbstractVector{<:Real}} = nothing)
    ctx_mat = build_seed_context(season; ctx = ctx, day_start = day_start, numeric_seed = numeric_seed)
    X = reshape(Float32.(ctx_mat), D_IN, ctx, 1)
    s0 = X[5, end, 1]
    c0 = X[6, end, 1]
    t_idx = nearest_slot_from_sc(s0, c0)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    pred_z = Matrix{Float32}(undef, D_OUT, n_slots)
    @inbounds for i in 1:n_slots
        t_idx_next = t_idx == SLOTS ? 1 : (t_idx + 1)
        xhat = integrate_cfm_midpoint(model, X, t_idx_next; steps = steps, rng = rng)
        pred_z[:, i] = xhat

        xin = X[:, end, 1]
        xin_next, t_idx = build_input_from_prediction(xin, xhat, t_idx)
        xin_next3 = reshape(xin_next, D_IN, 1, 1)
        X = cat(@view(X[:, 2:end, :]), xin_next3; dims = 2)
    end

    return pred_z
end

function generate_day_z(model::FMTransformer;
                        season::AbstractString = "Winter",
                        ctx::Int = CTX,
                        day_start::Time = Time(0, 0),
                        steps::Int = 38,
                        seed::Union{Nothing, Int} = nothing,
                        numeric_seed::Union{Nothing, AbstractVector{<:Real}} = nothing)
    return generate_slots_z(model;
                            n_slots = SLOTS,
                            season = season,
                            ctx = ctx,
                            day_start = day_start,
                            steps = steps,
                            seed = seed,
                            numeric_seed = numeric_seed)
end

function build_day_times(; day::Date = Date(2026, 1, 1), day_start::Time = Time(0, 0))
    start_dt = DateTime(day) + Hour(hour(day_start)) + Minute(minute(day_start))
    return [start_dt + Minute(10) * (i - 1) for i in 1:SLOTS]
end

function apply_value_formatting!(mat::AbstractMatrix)
    @inbounds for (j, feat) in enumerate(ORDERED_BASE)
        if feat in READABLE_FLOAT_FEATS
            mat[j, :] .= round.(Float64.(mat[j, :]); digits = 2)
        else
            mat[j, :] .= floor.(max.(Float64.(mat[j, :]), 0.0))
        end
    end
    return mat
end

function generate_day(model::FMTransformer;
                      season::AbstractString = "Winter",
                      serial::AbstractString = SERIAL_DEFAULT,
                      day::Date = Date(2026, 1, 1),
                      day_start::Time = Time(0, 0),
                      ctx::Int = CTX,
                      steps::Int = 38,
                      seed::Union{Nothing, Int} = nothing,
                      numeric_seed::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      curtailment_threshold::Union{Nothing, Real} = nothing,
                      free_power_feature::AbstractString = "power_avail_wind_mean_kw")
    pred_z = generate_day_z(model;
                            season = season,
                            ctx = ctx,
                            day_start = day_start,
                            steps = steps,
                            seed = seed,
                            numeric_seed = numeric_seed)
    pred_denorm = inverse_log1p_zscore(pred_z; serial = serial)
    pred_fmt = apply_value_formatting!(Float64.(pred_denorm))
    times = build_day_times(; day = day, day_start = day_start)

    df = DataFrame()
    df[!, "time"] = [string(t) * ".0" for t in times]
    @inbounds for (j, feat) in enumerate(ORDERED_BASE)
        if feat in READABLE_FLOAT_FEATS
            df[!, feat] = vec(pred_fmt[j, :])
        else
            df[!, feat] = Int.(vec(pred_fmt[j, :]))
        end
    end

    if !isnothing(curtailment_threshold)
        idx = findfirst(==(free_power_feature), ORDERED_BASE)
        isnothing(idx) && error("free_power_feature=$free_power_feature not in ORDERED_BASE")
        free_power = max.(0.0, Float64.(pred_fmt[idx, :]) .- Float64(curtailment_threshold))
        df[!, "curtailment_threshold"] = fill(Float64(curtailment_threshold), nrow(df))
        df[!, "free_power_kw"] = free_power
    end

    return (df = df, pred_z = pred_z, pred_denorm = pred_denorm)
end

function generate_day(; kwargs...)
    if !isdefined(@__MODULE__, :model)
        local_model = load_fm_model()
        return generate_day(local_model; kwargs...)
    end
    return generate_day(model; kwargs...)
end

function generate_days_all_seasons(model::FMTransformer;
                                   seasons = SEASON_NAMES,
                                   serial::AbstractString = SERIAL_DEFAULT,
                                   day::Date = Date(2026, 1, 1),
                                   day_start::Time = Time(0, 0),
                                   ctx::Int = CTX,
                                   steps::Int = 38,
                                   seed::Union{Nothing, Int} = nothing,
                                   numeric_seed::Union{Nothing, AbstractVector{<:Real}} = nothing,
                                   curtailment_threshold::Union{Nothing, Real} = nothing,
                                   free_power_feature::AbstractString = "power_avail_wind_mean_kw")
    out = Dict{String, Any}()
    for (i, s) in enumerate(seasons)
        season_seed = isnothing(seed) ? nothing : (seed + i - 1)
        out[String(s)] = generate_day(model;
                                      season = String(s),
                                      serial = serial,
                                      day = day,
                                      day_start = day_start,
                                      ctx = ctx,
                                      steps = steps,
                                      seed = season_seed,
                                      numeric_seed = numeric_seed,
                                      curtailment_threshold = curtailment_threshold,
                                      free_power_feature = free_power_feature)
    end
    return out
end

function generate_days_all_seasons(; kwargs...)
    if !isdefined(@__MODULE__, :model)
        local_model = load_fm_model()
        return generate_days_all_seasons(local_model; kwargs...)
    end
    return generate_days_all_seasons(model; kwargs...)
end
