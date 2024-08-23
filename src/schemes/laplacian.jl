using LinearAlgebra
using SparseArrays

include("boundary_conditions.jl")

struct RectangularDomain
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
end

# Based on: https://discourse.julialang.org/t/finite-difference-laplacian-with-five-point-stencil/25014
"""Matrix-free five-point Laplacian scheme with doubly periodic boundary conditions."""
function laplace_5p(u::Matrix{Float64}, dx::Float64)
    M, P = size(u)
    lap = zeros(M, P)

    @inbounds for j in 2:P-1
        @inbounds for i in 2:M-1
            lap[i, j] = (u[i-1, j] + u[i+1, j] - 4u[i, j] + u[i, j-1] + u[i, j+1]) * dx^-2
        end
    end

    update_doubly_periodic_bc!(lap)
    return lap
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
    # Negative so matrix is positive definite (for alpha != 0).
    A = -construct_spA(M, P, dx, alpha)
    return cholesky(A)
end

function get_poisson_cholesky(M::Int, P::Int, dx::Float64)
    # Negative so matrix is positive semi-definite.
    A = -construct_spA(M, P, dx, 0.0)

    # Ensure matrix is positive definite by fixing a point and making the kernel trivial.
    A[:,1] .= 0
    A[1,:] .= 0
    A[1, 1] = 1
    return cholesky(A)
end

"""Doubly periodic boundary conditions. Does not cache A factorisation so should only be used in single use cases."""
function sp_solve_modified_helmholtz(M::Int, P::Int, dx::Float64, f::Matrix{Float64}, alpha::Float64)
    chol_A = get_helmholtz_cholesky(M, P, dx, alpha)

    # Select the inner square of the domain to perform the solve on.
    b = -vec(copy(f[2:end-1, 2:end-1]))
    
    u = reshape(chol_A \ b, (M, P))
    return add_doubly_periodic_boundaries(u)
end

"""Solve the modified Helmholtz problem on a rectangular, doubly-periodic domain."""
function sp_solve_modified_helmholtz(M::Int, P::Int, dx::Float64, f_rhs::Function, alpha::Float64, domain::RectangularDomain)
    # Range includes extra index at each end for ghost cell.
    xs = range(domain.x1 - dx, domain.x2, length=M+2)
    ys = range(domain.y1 - dx, domain.y2, length=P+2)
    
    inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]
    b = inflate(f_rhs, xs, ys)
    
    return sp_solve_modified_helmholtz(M, P, dx, b, alpha)
end

function sp_solve_poisson(M::Int, P::Int, dx::Float64, f::Matrix{Float64})
    chol_A = get_poisson_cholesky(M, P, dx)

    # Select the inner square of the domain to perform the solve on.
    b = -vec(copy(f[2:end-1, 2:end-1]))
    
    # Reduce the number of unknowns to ensure system is positive definite.
    b[1] = 0

    u = reshape(chol_A \ b, (M, P))
    return add_doubly_periodic_boundaries(u)
end
