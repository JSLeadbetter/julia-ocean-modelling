using BenchmarkTools

include("helmholtz_dense.jl")
include("helmholtz_sparse.jl")

# Discrete points in our grid.
M = 64

# Modified helmholtz parameter.
alpha = 3.0

x1 = -1
x2 = 1
y1 = -1
y2 = 1


println("Constructing A")
@btime construct_A(M, alpha, dx)
@btime construct_spA(M, alpha, dx)

println("Constructing B")
@btime construct_B(M, alpha, dx)
@btime construct_spB(M, alpha, dx)

println("Solving for u")
@btime solve_modified_helmholtz(alpha, M, b_mp, x1, x2, y1, y2)
@btime sp_solve_modified_helmholtz(alpha, M, b_mp, x1, x2, y1, y2)