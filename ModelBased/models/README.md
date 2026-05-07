# ModelBased FM model runtime

This folder contains a local copy of the FM transformer assets from `../ESN4NW-Models`:

- `saves_HK_FM/1011089.jld2`
- `HK_blocks/stats_log1p_cache.json`
- `HK_blocks/stats_by_serial_year.json`
- `HK_blocks/stats_by_season_serial.json`
- `HK_model.jl` (model architecture + midpoint integrator)
- `FM_day_runtime.jl` (loading + one-day generation helpers)

## Quick start

```julia
include("ModelBased/models/FM_day_runtime.jl")

# Load model (also stored globally as `model`)
load_fm_model()

# Generate one day (00:00 -> 23:50) for one season with midpoint integration
res = generate_day(; season="Winter", day=Date(2026, 1, 1), steps=38)

# DataFrame with generated denormalized channels
df = res.df
first(df, 5)
```

## All seasons

```julia
all_days = generate_days_all_seasons(; day=Date(2026, 1, 1), steps=38)

# Example: summer day dataframe
summer_df = all_days["Sommer"].df
```

## Curtailment threshold + free available power

`generate_day` can also add `curtailment_threshold` and `free_power_kw`:

```julia
res = generate_day(;
    season="Winter",
    curtailment_threshold=400.0,
    free_power_feature="power_avail_wind_mean_kw"
)
```

`free_power_kw` is computed as:

`max(0.0, feature_value - curtailment_threshold)`
