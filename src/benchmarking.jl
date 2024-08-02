using BenchmarkTools

include("schemes/helmholtz.jl")

@btime begin
    M = P = 40
    alpha = 0.0
    dx = 1.0
    get_poisson_chol_A(M, P, dx)
end
