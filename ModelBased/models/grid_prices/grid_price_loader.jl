using CSV
using DataFrames
using Dates
using Random
using PlotlyJS

const GRID_PRICE_COL = "Deutschland/Luxemburg [€/MWh] Originalauflösungen"
const GRID_DATE_COL_FALLBACK = "Datum von"

const GRID_PRICE_FILES = Dict(
    "Winter" => joinpath(@__DIR__, "Gro_handelspreise_Winter.csv"),
    "Fruehling" => joinpath(@__DIR__, "Gro_handelspreise_Fruehling.csv"),
    "Sommer" => joinpath(@__DIR__, "Gro_handelspreise_Sommer.csv"),
    "Herbst" => joinpath(@__DIR__, "Gro_handelspreise_Herbst.csv"),
)

const _GRID_PRICE_DAY_CACHE = Dict{String, Dict{Date, Vector{Float32}}}()
const _GRID_PRICE_CANDIDATE_DAYS = Dict{String, Vector{Date}}()
const _GRID_PRICE_FIXED_DAY = Dict{String, Date}()
const GRID_PRICE_GLOBAL_NORM_FACTOR = Ref{Float32}(1f0)

_strip_bom(s::AbstractString) = replace(String(s), '\ufeff' => "")

function _canonical_season(season::AbstractString)
    s = lowercase(strip(String(season)))
    s = replace(s, "ä" => "ae", "ö" => "oe", "ü" => "ue")
    s = replace(s, "frühling" => "fruehling")
    if s == "winter"
        return "Winter"
    elseif s == "fruehling"
        return "Fruehling"
    elseif s == "sommer"
        return "Sommer"
    elseif s == "herbst"
        return "Herbst"
    end
    error("Unknown season '$season'. Supported: Winter, Fruehling, Sommer, Herbst.")
end

function _resolve_columns(df::DataFrame)
    names_raw = names(df)
    names_clean = _strip_bom.(String.(names_raw))

    date_idx = findfirst(==(GRID_DATE_COL_FALLBACK), names_clean)
    if isnothing(date_idx)
        date_idx = findfirst(n -> occursin("Datum von", n), names_clean)
    end

    price_idx = findfirst(==(GRID_PRICE_COL), names_clean)

    isnothing(date_idx) && error("Column '$GRID_DATE_COL_FALLBACK' not found in grid price CSV.")
    isnothing(price_idx) && error("Column '$GRID_PRICE_COL' not found in grid price CSV.")

    return names_raw[date_idx], names_raw[price_idx]
end

function _parse_datetime(raw)::Union{DateTime, Nothing}
    if raw === missing
        return nothing
    end
    s = strip(String(raw))
    isempty(s) && return nothing
    try
        return DateTime(s, dateformat"dd.mm.yyyy HH:MM")
    catch
        return nothing
    end
end

function _parse_price(raw)::Union{Float32, Nothing}
    if raw === missing
        return nothing
    elseif raw isa Number
        v = Float32(raw)
        return isfinite(v) ? v : nothing
    end

    s = strip(String(raw))
    isempty(s) && return nothing
    s == "-" && return nothing
    s = replace(s, "," => ".")
    try
        v = Float32(parse(Float64, s))
        return isfinite(v) ? v : nothing
    catch
        return nothing
    end
end

function _load_grid_price_cache!(season::AbstractString)
    haskey(_GRID_PRICE_DAY_CACHE, season) && return

    file = GRID_PRICE_FILES[season]
    isfile(file) || error("Grid price CSV not found: $file")

    df = CSV.read(file, DataFrame; delim = ';', decimal = ',', normalizenames = false)
    date_col, price_col = _resolve_columns(df)

    day_rows = Dict{Date, Vector{Tuple{Time, Float32}}}()
    for row in eachrow(df)
        dt = _parse_datetime(row[date_col])
        val = _parse_price(row[price_col])
        if isnothing(dt) || isnothing(val)
            continue
        end
        d = Date(dt)
        t = Time(dt)
        if !haskey(day_rows, d)
            day_rows[d] = Tuple{Time, Float32}[]
        end
        push!(day_rows[d], (t, val))
    end

    daily_series = Dict{Date, Vector{Float32}}()
    for (d, tv) in day_rows
        sort!(tv, by = x -> x[1])
        # keep only full regular days (96 quarter-hour rows)
        if length(tv) == 96
            daily_series[d] = Float32[x[2] for x in tv]
        end
    end

    # candidates require previous day availability for history pre-roll
    candidates = sort(Date[d for d in keys(daily_series) if haskey(daily_series, d - Day(1))])

    _GRID_PRICE_DAY_CACHE[season] = daily_series
    _GRID_PRICE_CANDIDATE_DAYS[season] = candidates
end

function _recompute_grid_price_global_norm_factor!()
    max_abs = 0f0
    for daily in values(_GRID_PRICE_DAY_CACHE)
        for series in values(daily)
            for v in series
                av = abs(Float32(v))
                if av > max_abs
                    max_abs = av
                end
            end
        end
    end
    GRID_PRICE_GLOBAL_NORM_FACTOR[] = max(max_abs, 1f-6)
    return GRID_PRICE_GLOBAL_NORM_FACTOR[]
end

function load_all_grid_price_caches!()
    for season in keys(GRID_PRICE_FILES)
        _load_grid_price_cache!(season)
    end
    return _recompute_grid_price_global_norm_factor!()
end

grid_price_norm_factor() = GRID_PRICE_GLOBAL_NORM_FACTOR[]

function interpolate_15min_to_5min(values_15::AbstractVector{<:Real})
    n = length(values_15)
    n > 0 || error("values_15 must not be empty.")

    out = Vector{Float32}(undef, 3 * n)
    @inbounds for i in 1:(n - 1)
        v0 = Float32(values_15[i])
        v1 = Float32(values_15[i + 1])
        k = 3 * (i - 1)
        out[k + 1] = v0
        out[k + 2] = (2f0 * v0 + v1) / 3f0
        out[k + 3] = (v0 + 2f0 * v1) / 3f0
    end

    v_last = Float32(values_15[end])
    k = 3 * (n - 1)
    out[k + 1] = v_last
    out[k + 2] = v_last
    out[k + 3] = v_last

    return out
end

function _gaussian_kernel(window::Int, sigma::Float64)
    window >= 1 || error("window must be >= 1")
    isodd(window) || error("window must be odd")
    sigma > 0 || error("sigma must be > 0")

    half = (window - 1) ÷ 2
    k = [exp(-0.5 * (i / sigma)^2) for i in -half:half]
    s = sum(k)
    return Float32.(k ./ s)
end

function smooth_grid_price(values::AbstractVector{<:Real};
                           window::Int = 9,
                           sigma::Float64 = 2.0,
                           passes::Int = 1)
    passes >= 0 || error("passes must be >= 0")
    x = Float32.(values)
    passes == 0 && return x

    kernel = _gaussian_kernel(window, sigma)
    half = (length(kernel) - 1) ÷ 2
    tmp = similar(x)

    for _ in 1:passes
        @inbounds for i in eachindex(x)
            acc = 0f0
            for j in eachindex(kernel)
                idx = clamp(i + (j - half - 1), firstindex(x), lastindex(x))
                acc += kernel[j] * x[idx]
            end
            tmp[i] = acc
        end
        x, tmp = tmp, x
    end

    return x
end

"""
    sample_grid_price_day(; season="Winter", history_steps=5, rng=Random.default_rng(),
                           apply_smoothing=true, smoothing_window=9, smoothing_sigma=2.0,
                           smoothing_passes=1, same_day=false)

Pick a random full day from the season CSV and return a Float32 vector with length
`288 + history_steps` (5-minute resolution). The first `history_steps` values come
from the end of the previous day.
"""
function sample_grid_price_day(; season::AbstractString = "Winter",
                               history_steps::Integer = 5,
                               rng::AbstractRNG = Random.default_rng(),
                               apply_smoothing::Bool = true,
                               smoothing_window::Int = 9,
                               smoothing_sigma::Float64 = 2.0,
                               smoothing_passes::Int = 1,
                               normalize_by_global_factor::Bool = true,
                               same_day::Bool = false)
    history_steps >= 0 || error("history_steps must be >= 0")

    season_key = _canonical_season(season)
    _load_grid_price_cache!(season_key)

    candidates = _GRID_PRICE_CANDIDATE_DAYS[season_key]
    isempty(candidates) && error("No valid day candidates found for season '$season_key'.")

    day = if same_day
        if !haskey(_GRID_PRICE_FIXED_DAY, season_key)
            _GRID_PRICE_FIXED_DAY[season_key] = first(candidates)
        end
        _GRID_PRICE_FIXED_DAY[season_key]
    else
        rand(rng, candidates)
    end
    prev_day = day - Day(1)

    y_prev_15 = _GRID_PRICE_DAY_CACHE[season_key][prev_day]
    y_day_15 = _GRID_PRICE_DAY_CACHE[season_key][day]

    y_prev_5 = interpolate_15min_to_5min(y_prev_15)
    y_day_5 = interpolate_15min_to_5min(y_day_15)

    history_steps <= length(y_prev_5) || error("history_steps=$history_steps exceeds previous-day 5-minute length=$(length(y_prev_5)).")

    out = Vector{Float32}(undef, length(y_day_5) + history_steps)  # 288 + history_steps
    if history_steps > 0
        out[1:history_steps] .= y_prev_5[end-history_steps+1:end]
    end
    out[history_steps+1:end] .= y_day_5

    if apply_smoothing
        out = smooth_grid_price(out;
                                window = smoothing_window,
                                sigma = smoothing_sigma,
                                passes = smoothing_passes)
    end

    if normalize_by_global_factor
        out ./= ( grid_price_norm_factor() / 4 )
        out .-= 0.25f0
        clamp!(out, 0f0, 1f0)
    end

    return out
end

# Initialize season caches + global normalization factor on script load.
load_all_grid_price_caches!()
