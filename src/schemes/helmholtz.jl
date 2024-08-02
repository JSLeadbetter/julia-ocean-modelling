using LinearAlgebra
using SparseArrays
using LinearSolve

# Boundary conditions
include("boundary_conditions.jl")

inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]
# speye(n::Int) = spdiagm(ones(n))

struct RectangularDomain
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
end

function laplacian_1d(N::Int)
    lap = spdiagm(-1 => ones(N-1), 0 => -2ones(N), 1 => ones(N-1))
    lap[1, end] = 1
    lap[end, 1] = 1
    return lap
end

function laplacian_2d(M::Int, P::Int)
    Dx = laplacian_1d(M)
    Dy = laplacian_1d(P)
    return kron(I(P), Dx) + kron(Dy, I(M))
end

function construct_spA(M::Int, P::Int, dx::Float64, alpha::Float64)
    A = (1 / dx^2) * laplacian_2d(M, P)
    A -= alpha * (dx^2) * I(M*P)
    return A
end

"""Solve the modified Helmholtz problem on a rectangular domain. Doubly periodic boundary conditions."""
function sp_solve_modified_helmholtz(M::Int, P::Int, dx::Float64, f::Function, alpha::Float64, domain::RectangularDomain)
    # Range includes extra index at each end for ghost cell.
    xs = range(domain.x1 - dx, domain.x2, length=M+2)
    ys = range(domain.y1 - dx, domain.y2, length=P+2)
    b = inflate(f, xs, ys)
    return sp_solve_modified_helmholtz(M, P, dx, b, alpha)
end

"""Solve the modified Helmholtz problem on a rectangular domain. Doubly periodic boundary conditions.
Does not cache LU factorisation so should only be used in single use cases."""
function sp_solve_modified_helmholtz(M::Int, P::Int, dx::Float64, f::Matrix{Float64}, alpha::Float64)
    A = get_helmholtz_chol_A(M, P, dx, alpha)
    
    # Select the inner square of the domain to perform the solve on.
    b = -vec(copy(f))
    b[1] = 0

    # # Ensure periodic boundary conditons are added.
    # update_doubly_periodic_bc!(f)

    # Solve the system.
    u = A \ b

    # u_re = reshape(u, (M, P))

    # u_extended = add_doubly_periodic_boundaries(u_re)

    return reshape(u, (M+2, P+2))
end

function add_doubly_periodic_boundaries(m::Matrix{Float64})
    M, P = size(m)
    new_m = zeros(M+2, P+2)
    new_m[2:end-1, 2:end-1] = m
    update_doubly_periodic_bc!(new_m)
    return new_m
end

function sp_solve_poisson(M::Int, P::Int, dx::Float64, f::Matrix{Float64})
    return sp_solve_modified_helmholtz(0.0, M, P, f, dx)
end

"""Initialise the Helmholtz problem with Cholesky factorisation completed."""
function get_helmholtz_chol_A(M::Int, P::Int, dx::Float64, alpha::Float64)
    # Use - here to ensure the matrix is pos def.
    A = -construct_spA(M+2, P+2, dx, alpha)

    A[:,1] .= 0
    A[1,:] .= 0
    A[1, 1] = 1
    
    cholesky!(A)
    return A
    
    # b = vec(zeros(M+2,P+2)) 

    # # Solve the system.
    # prob = LinearProblem(A, b)
    # linsolve = init(prob)
    # return linsolve
end

"""Initialise the Poisson problem with Cholesky factorisation completed."""
function get_poisson_chol_A(M::Int, P::Int, dx::Float64)
    return get_helmholtz_chol_A(M, P, dx, 0.0)
end
