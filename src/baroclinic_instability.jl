using LinearAlgebra
using Plots
using JLD

include("arakawa.jl")
include("helmholtz_sparse.jl")

"""Parameters for our ocean model."""
mutable struct BaroclinicModel
    H_1::Real # The height of the first layer in metres.
    H_2::Real # The height of the second layer in metres.
    H::Real # The total height of the model.
    beta::Real
    Lx::Real # The length of the rectangular domain.
    Ly::Real # The width of the rectangular domain.
    domain::RectangularDomain
    zeta_t::Real # Timestep we are on for our zeta grid.
    psi_t::Real # Timestep we are on for our psi grid.
    dt::Real # The timestep.
    T::Real # The time to stop at.
    U::Real # Forcing term of top level, speed of flow from left to right.
    M::Int # Number of nodes in the x direction.
    P::Int # Number of nodes in the y direction.
    dx::Real # Spacial step size.
    visc::Real # Viscosity.
    zeta::Matrix{Float64}    
    psi::Array{Float64, 3}
end

"""Outer constructor with default values."""
BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, zeta, psi) = BaroclinicModel(
    H_1, H_2, H_1+H_2, beta, Lx, Ly, RectangularDomain(0, Lx, 0, Ly), 0, dt, T, U, M, P, dx, visc, zeta, psi)


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
    2
    # Apply the Laplace operator using central differences
    for i in 2:M-1
        for j in 2:P-1
            lap_u[i, j] = (u[i-1, j] + u[i+1, j] - 4u[i, j] + u[i, j-1] + u[i, j+1]) / dx^2
        end
    end
    
    return lap_u
end

function evolve_zeta!(model::BaroclinicModel)
    # Define the function f used in the updates.
    f(zeta, psi, dx, beta, visc) = visc*laplace_5p(zeta, dx) - J(dx, zeta, psi) - beta*cd(psi, dx) - U*cd(zeta, dx)

    # First step or second step, we need to use Euler's method.
    t = model.t
    if t == 1
        # Euler's methed for the first step.
        f1 = f(model.zeta[:,:,1], model.psi[:,:,1], dx, beta, model.visc)
        new_zeta = model.dt * f1

        # Update our stored zetas.
        model.zeta[:,:,2] = model.zeta[:,:,1]
        model.zeta[:,:,1] = new_zeta
    elseif t == 2
        # Euler's method for the second step.
        f1 = f(model.zeta[:,:,1], model.psi[:,:,1], dx, beta, model.visc)
        new_zeta = model.dt * f1
        
        # Update our stored zetas.
        model.zeta[:,:,3] = model.zeta[:,:,2]
        model.zeta[:,:,2] = model.zeta[:,:,1]
        model.zeta[:,:,1] = new_zeta
    else
        # AB3 for subsequent steps.
        # h*((23/12)*f(t[i+2], u[i+2]) - (16/12)*f(t[i+1], u[i+1]) + (5/12)*f(t[i], u[i]))
        f1 = f(model.zeta[:,:,1], model.psi[:,:,1], dx, beta, model.visc)
        f2 = f(model.zeta[:,:,2], model.psi[:,:,2], dx, beta, model.visc)
        f3 = f(model.zeta[:,:,3], model.psi[:,:,3], dx, beta, model.visc)
        new_zeta = model.dt * ((23/12)*f1 - (16/12)*f2 + (5/12)*f3) 
        
        # Update our variables with the new 
        model.zeta[:,:,3] = model.zeta[:,:,2]
        model.zeta[:,:,2] = model.zeta[:,:,1]
        model.zeta[:,:,1] = new_zeta
    end
end

"""Evolve the potential velocity field for one level of the system."""
function evolve_zeta(zeta::Matrix{Float64}, psi::Matrix{Float64}, dx::Float64, beta::Float64, visc::Float64, dt::Float64, U::Float64, t::Int)
    
    

    f(zeta, psi, dx, beta, visc) = visc*laplace_5p(zeta, dx) - J(dx, zeta, psi) - beta*cd(psi, dx) - U*cd(zeta, dx)
    
    if 
    
    
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

function evolve_psi(model::BaroclinicModel)
    f_rhs = P_inv_matrix(model.H_1, model.H_2) * model.zeta

    # 1. Solve the Poisson problem.
    u_1 = sp_solve_modified_helmholtz(0, model.M, model.P, f_rhs, model.dx, model.domain)
    model.psi[:,:,2] = u_1

    # 2. Solve the modified Helmholtz problem.
    lambda = -(model.H) / (model.H_1 * model.H_2)

    u_2 = sp_solve_modified_helmholtz(lambda, model.M, model.P, f_rhs, model.dx, model.domain)
    model.psi[:,:,2] = u_2
end

function main()
    H_1 = 1000
    H_2 = 2000
    beta = 2*10^-11
    Lx = 4000 # 4000 km
    Ly = 2000 # 2000 km
    dt = 1 # 30 minutes
    T = 50
    U = 1.0 # Forcing term of top level.
    M = 100
    P = 100
    dx = 1.0
    # beta = 1.0
    visc = 1.0
    zeta = rand(M, P, T)
    psi = rand(M, P, T)

    # Create the model.
    model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, zeta, psi)

    # velocity = zeros(M, P, T, 2)

    for i in 1:T-1



        evolve_zeta(mode)
        # First solve for the evolution of zeta.
        zeta[:,:,i+1] = evolve_zeta(zeta[:,:,i], psi[:,:,i], dx, beta, visc, dt, U)

        # Then use the latest value of zeta to solve for the latest value of psi.
        # psi[:,:,i+1] = evolve_phi(zeta[:,:,i+1], psi[:,:,i])

        evolve_psi(model)

        # Calculate the other quantities for plotting.
        # velocity[:,:,i+1,:] = velocity_matrix(psi[:,:,i+1], dx)
    end

    # Saving the data at the end of the simulation.
    save("data/zeta.jld", "zeta", zeta)
    save("data/psi.jld", "psi", psi)
    save("data/velocity.jld", "velocity", velocity)

    return 0
end

main()
