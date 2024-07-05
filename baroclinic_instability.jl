using LinearAlgebra
using Plots
using JLD

include("arakawa.jl")
include("helmholtz_sparse.jl")

"""
Centred-difference scheme for matrices in the x-direction.

Uses periodic boundary conditions with 0 at the boundary.
"""
function cd(u::Matrix{Float64}, dx::Float64)
    M, P = size(u)
    u_out = zeros(M, P)
    
    for i in 2:M-1
        for j in 2:P-1
            u_out[i, j] = (1/2dx)*(u[i+1] - u[i-1])
        end
    end

    return u_out
end

"""
Five-point Laplacian scheme.
"""
function laplace_5p(u::Matrix{Float64}, dx::Float64)
    M, P = size(u)
    lap_u = zeros(M, P)
    
    # Apply the Laplace operator using central differences
    for i in 2:M-1
        for j in 2:P-1
            lap_u[i, j] = (u[i-1, j] + u[i+1, j] - 4u[i, j] + u[i, j-1] + u[i, j+1]) / dx^2
        end
    end
    
    return lap_u
end

"""
Evolve the potential velocity field for one level of the system.
"""
function evolve_zeta(zeta::Matrix{Float64}, psi::Matrix{Float64}, dx::Float64, beta::Float64, visc::Float64, dt::Float64, U::Float64)
    f(zeta, psi, dx, beta, visc) = visc*laplace_5p(zeta, dx) - J(dx, zeta, psi) - beta*cd(psi, dx) - U*cd(zeta, dx)
    zeta += dt * f(zeta, psi, dx, beta, visc)
    return zeta
end

"Non-zero eigenvalue of the S matrix."
function S_eig(H_1::Float64, H_2::Float64)
    return -(H_1 + H_2) / (H_1 * H_2)
end

"The matrix of eigenvectors of the S matrix."
function P_matrix(H_1::Float64, H_2::Float64)
    P = ones(2, 2)
    P[1, 2] = -H_2/H_1
    return P
end

"The inverse of the matrix of eigenvectors of the P matrix."
function P_inv_matrix(H_1::Float64, H_2::Float64)
    H = H_1 + H_2
    P = zeros(2, 2)
    P[1, 1] = H_1 / H
    P[1, 2] = H_2 / H
    P[2, 1] = -P[1, 1]
    P[2, 2] = P[1, 1]
    return P
end

"""Calculate the velocity field using the streamfunction psi."""
function velocity_matrix(psi::Matrix{Float64}, dx::Float64)
    M, P = size(psi)
    u = zeros(M, P, 2)
    
    for i in 2:M-1
        for j in 2:P-1
            # Finite difference in the y direction.
            u[i, j, 1] = -(1/2dx)*(psi[i, j+1] - psi[i, j-1])

            # Finite difference in the x direction.
            u[i, j, 2] = (1/2dx)*(psi[i+1, j] - psi[i-1, j])
        end
    end

    return u
end

"""
"""
function evolve_psi(zeta, psi)
    
    # 1. Solve the Laplace problem.

    # 2. Solve the modified Helmholtz problem.


    # sp_solve_modified_helmholtz(alpha, M    )
    
    return psi
end

function main()
    H_1 = 1000
    H_2 = 2000
    beta = 2*10^-11
    Lx = 4000 # 4000 km
    Ly = 2000 # 2000 km

    # 30 minutes
    dt = 1

    T = 50

    # Forcing term of top level.
    U = 1.0

    M = 100
    P = 100
    dx = 1.0
    beta = 1.0
    visc = 1.0

    zeta = rand(M, P, T)
    psi = rand(M, P, T)
    velocity = zeros(M, P, T, 2)

    for i in 1:T-1
        # First solve for the evolution of zeta.
        zeta[:,:,i+1] = evolve_zeta(zeta[:,:,i], psi[:,:,i], dx, beta, visc, dt, U)

        # Then use the latest value of zeta to solve for the latest value of psi.
        psi[:,:,i+1] = evolve_phi(zeta[:,:,i+1], psi[:,:,i])

        velocity[:,:,i+1,:] = velocity_matrix(psi[:,:,i+1], dx)
    end

    # Saving the data at the end of the simulation.
    save("data/zeta.jld", "zeta", zeta)
    save("data/psi.jld", "psi", psi)
    save("data/velocity.jld", "velocity", velocity)

    return 0
end

main()
