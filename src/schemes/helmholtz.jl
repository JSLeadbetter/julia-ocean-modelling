using LinearAlgebra
using SparseArrays
using LinearSolve

# Boundary conditions
include("boundary_conditions.jl")

inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

struct RectangularDomain
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
end

# https://nbviewer.org/github/mitmath/18S096/blob/409bf1c1cbc8ed0f70afeb0f885ddc382f5138be/lectures/other/Nested-Dissection.ipynb
"""Rectangular domain M x P, dx varies."""
function construct_spA(M::Int, P::Int, alpha::Float64, dx::Float64)
    D1 = spdiagm(M+1, M, -1 => -ones(M), 0 => ones(M))
    D2 = spdiagm(P+1, P, -1 => -ones(P), 0 => ones(P))
    A = kron(speye(P), D1' * D1) + kron(D2' * D2, speye(M)) - (alpha * (dx^2))*speye(M*P)
    # TODO: 
    return A * (1 / (dx)^2)
end

"""Solve the modified Helmholtz problem on a rectangular domain.

Doubly periodic boundary conditions."""
function sp_solve_modified_helmholtz(alpha::Float64, M::Int, P::Int, f::Function, dx::Float64, domain::RectangularDomain)
    # Range includes extra index at each end for ghost cell.
    xs = range(domain.x1 - dx, domain.x2, length=M+2)
    ys = range(domain.y1 - dx, domain.y2, length=P+2)
    
    # Use - here to ensure the matrix is semi pos def.
    A = -construct_spA(M+2, P+2, alpha, dx)
    b = inflate(f, xs, ys)

    return sp_solve_modified_helmholtz(alpha, M, P, b, dx, domain)
end

"""Solve the modified Helmholtz problem on a rectangular domain. Doubly periodic boundary conditions.
Does not cache LU factorisation so should only be used in single use cases."""
function sp_solve_modified_helmholtz(alpha::Float64, M::Int, P::Int, f::Matrix{Float64}, dx::Float64)
    # Use - here to ensure the matrix is semi pos def.
    A = -construct_spA(M+2, P+2, alpha, dx)
    b = -vec(f)

    # Ensure periodic boundary conditons are added.
    update_doubly_periodic_bc!(f)

    # Solve the system.
    prob = LinearProblem(A, b)
    linsolve = init(prob)
    return solve(linsolve).u
end

function sp_solve_poisson(M::Int, P::Int, f::Matrix{Float64}, dx::Float64)
    return sp_solve_modified_helmholtz(0.0, M, P, f, dx)
end

"""Initialise the Helmholtz problem with LU factorisation completed."""
function init_sp_modified_helmholtz(alpha::Float64, M::Int, P::Int, dx::Float64)
    # Use - here to ensure the matrix is semi pos def.
    A = -construct_spA(M+2, P+2, alpha, dx)
    b = vec(zeros(M+2,P+2)) 

    # Solve the system.
    prob = LinearProblem(A, b)
    linsolve = init(prob)
    return linsolve
end

"""Initialise the Poisson problem with LU factorisation completed."""
function init_sp_poisson(M::Int, P::Int, dx::Float64)
    return init_sp_modified_helmholtz(0.0, M, P, dx)
end
