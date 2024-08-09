using LinearAlgebra
using SparseArrays
using LinearSolve
using SuiteSparse

# Boundary conditions
include("boundary_conditions.jl")

inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

struct RectangularDomain
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
end

function laplacian_1d(N::Int)
    return spdiagm(-1 => ones(N-1), 0 => -2ones(N), 1 => ones(N-1))
end

function laplacian_2d(M::Int, P::Int)
    Dx = laplacian_1d(M)
    Dy = laplacian_1d(P)
    return kron(I(P), Dx) + kron(Dy, I(M))
end

function laplacian_1d_periodic(N::Int)
    lap = laplacian_1d(N)
    lap[1, end] = 1
    lap[end, 1] = 1
    return lap
end

function laplacian_2d_doubly_periodic(M::Int, P::Int)
    Dx = laplacian_1d_periodic(M)
    Dy = laplacian_1d_periodic(P)
    return kron(I(P), Dx) + kron(Dy, I(M))
end

"""Matrix A in solving modified Helmholtz systems."""
function construct_spA(M::Int, P::Int, dx::Float64, alpha::Float64)
    A = laplacian_2d_doubly_periodic(M, P)
    A += alpha * dx^2 * I(M*P)
    return dx^-2 * A
end

function get_helmholtz_cholesky(M::Int, P::Int, dx::Float64, alpha::Float64)
    # Negative so matrix is positive semi-definite.
    A = -construct_spA(M, P, dx, alpha)

    # Ensure matrix is positive definite by reducing number of unknowns.
    A[:,1] .= 0
    A[1,:] .= 0
    A[1, 1] = 1

    return cholesky(A)
end

function get_poisson_cholesky(M::Int, P::Int, dx::Float64)
    return get_helmholtz_cholesky(M, P, dx, 0.0)
end

"""Initialise the Helmholtz problem."""
function get_helmholtz_linsolve_A(M::Int, P::Int, dx::Float64, alpha::Float64)
    # Negative so matrix is positive semi-definite.
    A = -construct_spA(M, P, dx, alpha)

    # Ensure matrix is positive definite by reducing number of unknowns.
    A[:,1] .= 0
    A[1,:] .= 0
    A[1, 1] = 1

    # Start with zeros so we can initialise the linear problem, could be any vector.
    b = vec(zeros(M,P)) 
    prob = LinearProblem(A, b)
    return init(prob)
end

function get_cholesky_factorisation(M::Int, P::Int, dx::Float64, alpha::Float64)
    # Negative so matrix is positive semi-definite.
    A = -construct_spA(M, P, dx, alpha)

    # Ensure matrix is positive definite by reducing number of unknowns.
    A[:,1] .= 0
    A[1,:] .= 0
    A[1, 1] = 1

    @assert typeof(A) == SparseMatrixCSC
    factorization = cholesky(A)#LinearSolve.factorize(A)
    return factorization
end

"""Initialise the Poisson problem."""
function get_poisson_linsolve_A(M::Int, P::Int, dx::Float64)
    return get_helmholtz_linsolve_A(M, P, dx, 0.0)
end

"""Extend a matrix by two rows and columns and copies rows/cols to add double periodicity."""
function add_doubly_periodic_boundaries(u::Matrix{Float64})
    M, P = size(u)
    extended_u = zeros(M+2, P+2)
    extended_u[2:end-1, 2:end-1] = u
    update_doubly_periodic_bc!(extended_u)
    return extended_u
end

"""Doubly periodic boundary conditions. Does not cache A factorisation so should only be used in single use cases."""
function sp_solve_modified_helmholtz(M::Int, P::Int, dx::Float64, f::Matrix{Float64}, alpha::Float64)
    linsolve = get_helmholtz_linsolve_A(M, P, dx, alpha)

    # Select the inner square of the domain to perform the solve on.
    b = -vec(copy(f[2:end-1, 2:end-1]))
    
    # Reduce the number of unknowns to ensure system is positive definite.
    b[1] = 0

    linsolve.b = b
    u = reshape(solve(linsolve).u, (M, P))
    return add_doubly_periodic_boundaries(u)
end

"""Solve the modified Helmholtz problem on a rectangular domain. Doubly periodic boundary conditions."""
function sp_solve_modified_helmholtz(M::Int, P::Int, dx::Float64, f::Function, alpha::Float64, domain::RectangularDomain)
    # Range includes extra index at each end for ghost cell.
    xs = range(domain.x1 - dx, domain.x2, length=M+2)
    ys = range(domain.y1 - dx, domain.y2, length=P+2)
    b = inflate(f, xs, ys)
    return sp_solve_modified_helmholtz(M, P, dx, b, alpha)
end

function sp_solve_poisson(M::Int, P::Int, dx::Float64, f::Matrix{Float64})
    return sp_solve_modified_helmholtz(M, P, dx, f, alpha)
end
