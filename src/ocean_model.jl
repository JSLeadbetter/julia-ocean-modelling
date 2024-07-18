using LinearAlgebra
using Plots
using JLD

include("schemes/arakawa.jl")
include("schemes/helmholtz_sparse.jl")

"""Parameters for our ocean model."""
struct BaroclinicModel
    H_1::Float64 # The height of the first layer in metres.
    H_2::Float64 # The height of the second layer in metres.
    H::Float64 # The total height of the model.
    beta::Float64
    Lx::Float64 # The length of the rectangular domain.
    Ly::Float64 # The width of the rectangular domain.
    domain::RectangularDomain
    dt::Float64 # The timestep.
    T::Float64 # The time to stop at.
    U::Float64 # Forcing term of top level, speed of flow from left to right.
    M::Int # Number of nodes in the x direction.
    P::Int # Number of nodes in the y direction.
    dx::Float64 # Spacial step size.
    visc::Float64 # Viscosity.
end

"""Outer constructor with default values."""
BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc) = BaroclinicModel(
    H_1, H_2, H_1+H_2, beta, Lx, Ly, RectangularDomain(0, Lx, 0, Ly), dt, T, U, M, P, dx, visc)


"""Centred-difference scheme for matrices in the x-direction.

Uses periodic boundary conditions with 0 at the boundary."""
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

# https://discourse.julialang.org/t/finite-difference-laplacian-with-five-point-stencil/25014
"""Five-point Laplacian scheme."""
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

function store_new_state(arr::Array{Float64, 4}, new_state::Matrix{Float64}, z::Int)
    arr[:,:,z,3] = arr[:,:,z,2]
    arr[:,:,z,2] = arr[:,:,z,1]
    arr[:,:,z,1] = new_state
    return arr
end

function evolve_zeta(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, timestep::Int)
    for z = 1:2
        # TODO: Implement the different formulae for layer 1 and 2.
        
        # Define the function f used in the updates. Begin ... end used for clean multiline declaration.
        f(zeta::Matrix{Float64}, psi::Matrix{Float64}) = begin
            model.visc*laplace_5p(zeta, model.dx)
            - J(model.dx, zeta, psi)
            - model.beta*cd(psi, model.dx)
            - model.U*cd(zeta, model.dx)
        end

        # TODO: Pass zeta and psi in separately?

        if timestep == 1 || timestep == 2
            # Euler's methed for the first and second step.
            f1 = f(zeta[:,:,z,1], psi[:,:,z,1])
            new_zeta = zeta[:,:,z,1] + model.dt * f1
            new_zeta = update_ghost_cells(new_zeta)
        else
            # AB3 for subsequent steps.
            f1 = f(zeta[:,:,z,1], psi[:,:,z,1])
            f2 = f(zeta[:,:,z,2], psi[:,:,z,2])
            f3 = f(zeta[:,:,z,3], psi[:,:,z,3])
            new_zeta = zeta[:,:,z,1] + (model.dt * ((23/12)*f1 - (16/12)*f2 + (5/12)*f3))

            new_zeta = update_ghost_cells(new_zeta)
        end
        zeta = store_new_state(zeta, new_zeta, z)
    end
    return zeta
end

function evolve_psi(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4})
    # 1. Solve the Poisson problem for a.
    f_1 = zeta[:,:,1,1] - (model.H_2 / model.H_1) * zeta[:,:,2,1]
    a = sp_solve_modified_helmholtz(0.0, model.M, model.P, f_1, model.dx)
    a_re = reshape(a, (model.M+2, model.P+2))
    
    # 2. Solve the modified Helmholtz problem for b

    # The non-zero eigenvalue of our problem.
    lambda = -(model.H) / (model.H_1 * model.H_2)

    f_2 = zeta[:,:,1,1] + zeta[:,:,2,1] 
    b = sp_solve_modified_helmholtz(lambda, model.M, model.P, f_2, model.dx)
    b_re = reshape(b, (model.M+2, model.P+2))

    # Solve for psi_1 and psi_
    new_psi_1 = model.H_1*a_re
    new_psi_1 += model.H_2*b_re
    new_psi_1 = new_psi_1 / model.H
    
    new_psi_2 = (b_re - a_re) / ((model.H_2 / model.H_1) + 1)
    
    psi = store_new_state(psi, new_psi_1, 1)
    psi = store_new_state(psi, new_psi_2, 2)
    return psi
end
