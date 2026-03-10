using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "RL"))
Pkg.instantiate()
