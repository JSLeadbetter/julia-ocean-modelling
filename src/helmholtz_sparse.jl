using LinearAlgebra
using SparseArrays
using IterativeSolvers
using LinearSolve

include("toolbox.jl")

struct RectangularDomain
    x1::Real
    x2::Real
    y1::Real
    y2::Real
end

# https://nbviewer.org/github/mitmath/18S096/blob/409bf1c1cbc8ed0f70afeb0f885ddc382f5138be/lectures/other/Nested-Dissection.ipynb
"""Rectangular domain M x P, dx varies."""
function construct_spA(M::Int, P::Int, alpha::Real, dx::Real)
    D1 = spdiagm(M+1, M, -1 => -ones(M), 0 => ones(M))
    D2 = spdiagm(P+1, P, -1 => -ones(P), 0 => ones(P))
    A = kron(speye(P), D1' * D1) + kron(D2' * D2, speye(M)) - (alpha * (dx^2))*speye(M*P)
    return A * 1 / (dx)^2
end

"""Solve the modified Helmholtz problem on a rectangular domain."""
function sp_solve_modified_helmholtz(alpha::Real, M::Int, P::Int, f::Function, dx::Real, domain::RectangularDomain)
    xs = range(domain.x1 + dx, domain.x2 - dx, length=M-1)
    ys = range(domain.y1 + dx, domain.y2 - dx, length=P-1)
    
    # Use - here to ensure the matrix is semi pos def.
    A = -construct_spA(M-1, P-1, alpha, dx)
    b = vec(inflate(f, xs, ys))

    # Solve the system.
    prob = LinearProblem(A, b)
    linsolve = init(prob)
    return A, b, solve(linsolve).u
end

# A matrix B which is used as blocks inside the matrix A.
# function construct_spB(M::Int, alpha::Float64, dx::Float64)
#     diag_vals = diag_vals = (-4 + (alpha * dx^2)) * ones(M-1)
#     B = spdiagm(0 => diag_vals, 1 => ones(M-2), -1 => ones(M-2))
# 	return B
# end
# 
# function construct_spA(M::Int, alpha::Float64, dx::Float64)
# 	B = construct_spB(M, alpha, dx)
#   A = kron(speye(M-1), B) + kron(spdiagm(1 => ones(M-2), -1 => ones(M-2)), speye(M-1))
# 	return A * 1 / (dx)^2
# end

# """Rectangular domain M x P, dx = 1."""
# function construct_spA(M::Int, P::Int, alpha::Real)
#     D1 = spdiagm(M+1, M, -1 => -ones(M), 0 => ones(M))
#     D2 = spdiagm(P+1, P, -1 => -ones(P), 0 => ones(P))
#     return kron(speye(P), D1' * D1) + kron(D2' * D2, speye(M)) - alpha*speye(M*P)
# end 

# function sp_solve_modified_helmholtz(alpha::Float64, M::Int64, f::Matrix{Float64}, dx::Float64)
#     # Use - here to ensure the matrix is semi pos def.
#     A = -construct_spA(M, M, alpha, dx)
#     b = vec(f)
#     # Solve the system.
#     prob = LinearProblem(A, b)
#     linsolve = init(prob)
#     return solve(linsolve).u
# end

# """Solve the modified Helmholtz problem on a block domain of size M x P with dx = 1."""
# function sp_solve_modified_helmholtz(alpha::Real, M::Int, P::Int, f::Function)
#     dx = 1
#     xs = range(dx, M - dx, length=M-1)
#     ys = range(dx, P - dx, length=P-1)
    
#     # Use - here to ensure the matrix is semi pos def.
#     A = -construct_spA(M-1, P-1, alpha)
#     b = vec(inflate(f, xs, ys))

#     # Solve the system.
#     prob = LinearProblem(A, b)
#     linsolve = init(prob)
#     return A, b, solve(linsolve).u
# end

# function sp_solve_poisson(M::Int, P::Int, f::Function, dx::Real, domain::RectangularDomain)
#     xs = range(domain.x1 + dx, domain.x2 - dx, length=M-1)
#     ys = range(domain.y1 + dx, domain.y2 - dx, length=P-1)
    
#     # Use -A here to ensure the matrix is semi pos def.
#     A = -construct_spA(M-1, P-1, 0, dx)
#     b = vec(inflate(f, xs, ys))

#     # Solve the system.
#     prob = LinearProblem(A, b)
#     linsolve = init(prob)
#     return A, b, solve(linsolve).u
# end
