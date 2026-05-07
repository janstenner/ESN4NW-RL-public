using Dates
using PlotlyJS
using JSON
using Printf

include(joinpath(@__DIR__, "FM_day_runtime.jl"))

const SEASON = "Winter"
const FM_STEPS = 38
const FM_SEED = nothing
const FM_SERIAL = "1011089"
const STATS_SEASON_PATH = joinpath(@__DIR__, "HK_blocks", "stats_by_season_serial.json")
const STATS_YEAR_PATH = joinpath(@__DIR__, "HK_blocks", "stats_by_serial_year.json")

feature_index(name::AbstractString) = begin
    idx = findfirst(==(name), ORDERED_BASE)
    isnothing(idx) && error("Feature $name not found in ORDERED_BASE")
    idx
end

function build_time_axis_5min()
    return [string(Time(0, 0) + Minute(5) * (i - 1)) for i in 1:(2 * SLOTS)]
end

function interpolate_10min_to_5min(y10::AbstractVector{<:Real})
    n10 = length(y10)
    y5 = Vector{Float64}(undef, 2 * n10)

    @inbounds for k in 1:length(y5)
        if isodd(k)
            y5[k] = Float64(y10[(k + 1) ÷ 2])
        else
            idx = k ÷ 2
            if idx < n10
                y5[k] = 0.5 * (Float64(y10[idx]) + Float64(y10[idx + 1]))
            else
                # Last 5-minute point (23:55) gets held constant from 23:50.
                y5[k] = Float64(y10[end])
            end
        end
    end

    return y5
end

function load_stats()
    stats_season = JSON.parsefile(STATS_SEASON_PATH)
    stats_year = JSON.parsefile(STATS_YEAR_PATH)
    return stats_season, stats_year
end

function stat_mu_sigma(feat::AbstractString; season::AbstractString, serial::AbstractString, stats_season, stats_year)
    serial_key = String(serial)
    season_key = String(season)

    if haskey(stats_season, season_key) &&
       haskey(stats_season[season_key], serial_key) &&
       haskey(stats_season[season_key][serial_key], feat)
        obj = stats_season[season_key][serial_key][feat]
        return Float64(obj["mu"]), Float64(obj["sigma"])
    end

    if haskey(stats_year, serial_key) && haskey(stats_year[serial_key], feat)
        obj = stats_year[serial_key][feat]
        return Float64(obj["mu"]), Float64(obj["sigma"])
    end

    error("No stats found for feature=$feat, season=$season_key, serial=$serial_key")
end

function one_level_from_stats(feat::AbstractString; season::AbstractString, serial::AbstractString, stats_season, stats_year)
    mu, sigma = stat_mu_sigma(feat; season = season, serial = serial, stats_season = stats_season, stats_year = stats_year)
    return max(mu + 3.0 * sigma, 1e-6)
end

renorm01(signal::AbstractVector{<:Real}, one_level::Real) = Float64.(signal) ./ Float64(one_level)

function print_one_levels(one_levels::Dict{String, Float64})
    println("Denormalized value represented by 1.0:")
    for key in sort(collect(keys(one_levels)))
        @printf("  %-22s -> %.6f\n", key, one_levels[key])
    end
end

function main(; season::AbstractString = SEASON, steps::Int = FM_STEPS, seed = FM_SEED, serial::AbstractString = FM_SERIAL)

    seed = isnothing(seed) ? Int(floor(rand() * 1000000000)) : seed

    model = load_fm_model(; serial = serial)

    # One-day generation in model normalized space (00:00 -> 23:50).
    pred_z = generate_day_z(model; season = season, steps = steps, seed = seed)

    # Keep full denormalization path available.
    pred_denorm = inverse_log1p_zscore(pred_z; serial = serial)

    idx_power_avail_tech = feature_index("power_avail_tech_mean_kw")
    idx_power_avail_force = feature_index("power_avail_force_maj_mean_kw")
    idx_power_avail_ext = feature_index("power_avail_ext_mean_kw")

    tech = vec(pred_denorm[idx_power_avail_tech, :])
    force = vec(pred_denorm[idx_power_avail_force, :])
    ext = vec(pred_denorm[idx_power_avail_ext, :])

    wind_power = min.(tech, force)
    curtailment_threshold = ext

    stats_season, stats_year = load_stats()
    tech_one = one_level_from_stats("power_avail_tech_mean_kw"; season = season, serial = serial, stats_season = stats_season, stats_year = stats_year)
    force_one = one_level_from_stats("power_avail_force_maj_mean_kw"; season = season, serial = serial, stats_season = stats_season, stats_year = stats_year)

    wind_power_one = min(tech_one, force_one)
    common_one = wind_power_one

    one_levels = Dict(
        "wind_power" => common_one,
        "free_energy" => common_one,
        "curtailment_threshold" => common_one,
    )
    print_one_levels(one_levels)

    wind_power_01 = renorm01(wind_power, common_one)
    curtailment_threshold_01 = renorm01(curtailment_threshold, common_one)
    free_energy_10 = max.(wind_power_01 .- curtailment_threshold_01, 0.0)

    # 10-minute -> 5-minute interpolation.
    wind_power_5 = interpolate_10min_to_5min(wind_power_01)
    curtailment_threshold_5 = interpolate_10min_to_5min(curtailment_threshold_01)
    free_energy_5 = max.(wind_power_5 .- curtailment_threshold_5, 0.0)

    x = build_time_axis_5min()

    traces = [
        scatter(x = x, y = wind_power_5, mode = "lines", name = "wind_power"),
        scatter(x = x, y = free_energy_5, mode = "lines", name = "free_energy"),
        scatter(x = x, y = curtailment_threshold_5, mode = "lines", name = "curtailment_threshold"),
    ]

    layout = Layout(
        title = "FM Day Test - $(season) (renormalized + 5min)",
        xaxis = attr(title = "Time of day"),
        yaxis = attr(title = "Renormalized value"),
        plot_bgcolor = "#f1f3f7"
    )

    display(plot(traces, layout))

    return (
        x = x,
        pred_z = pred_z,
        pred_denorm = pred_denorm,
        wind_power_10 = wind_power_01,
        free_energy_10 = free_energy_10,
        curtailment_threshold_10 = curtailment_threshold_01,
        wind_power_5 = wind_power_5,
        free_energy_5 = free_energy_5,
        curtailment_threshold_5 = curtailment_threshold_5,
        one_levels = one_levels,
    )
end

result = main();
