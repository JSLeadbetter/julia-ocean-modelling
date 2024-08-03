using BenchmarkTools

include("schemes/helmholtz.jl")

@btime begin
    M = P = 20
    alpha = 1.0
    dx = 0.5
    A = get_helmholtz_chol_A(M, P, dx, alpha)
    b = vec(ones(M+2,P+2))
    A \ b 
end

@btime begin
    M = P = 20
    alpha = 1.0
    dx = 0.5
    linsolve = get_helmholtz_linsolve_A(M, P, dx, alpha)
    linsolve.b = vec(ones(M+2,P+2))
    solve(linsolve)
end

