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
    visc::Float64 # Viscosity term for Laplacian friction.
    r::Float64 # Bottom friction viscosity.
    R_d::Float64 # Deformation radius.
end

"""Outer constructor with default values."""
BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d) = BaroclinicModel(
    H_1, H_2, H_1+H_2, beta, Lx, Ly, RectangularDomain(0, Lx, 0, Ly), dt, T, U, M, P, dx, visc, r, R_d)


"""Centred-difference scheme for matrices in the x-direction.

TODO: Add periodic boundary conditions here
Uses periodic boundary conditions with 0 at the boundary."""
function cd(u::Matrix{Float64}, dx::Float64)
    M, P = size(u)
    u_out = zeros(M, P)
    
    for j in 2:P-1
        for i in 2:M-1
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
    for j in 2:P-1
        for i in 2:M-1
            lap_u[i, j] = (u[i-1, j] + u[i+1, j] - 4u[i, j] + u[i, j-1] + u[i, j+1]) / dx^2
        end
    end
    
    return lap_u
end

"The matrix of eigenvectors of the S matrix."
function P_matrix(H_1::Float64, H_2::Float64)
    P = ones(Float64, 2, 2)
    P[2, 2] = -H_2 / H_1
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

function store_new_state(arr::Array{Float64, 4}, new_state::Matrix{Float64}, z::Int)
    arr[:,:,z,3] = arr[:,:,z,2]
    arr[:,:,z,2] = arr[:,:,z,1]
    arr[:,:,z,1] = new_state
    return arr
end

"""(f_0/N_0)^2"""
function ratio_term(model::BaroclinicModel)
    return ((model.H_1 + model.H_2) / (2*model.R_d^2)) * ((1/model.H_1) + (1/model.H_2))
end

function S1_plus(model::BaroclinicModel)
    return (2 * ratio_term(model)) / (model.H_1 * (model.H_1 + model.H_2))
end

function S2_minus(model::BaroclinicModel)
    return (2 * ratio_term(model)) / (model.H_2 * (model.H_1 + model.H_2))
end

# Modified beta functions.
beta_1(model::BaroclinicModel) = model.beta + S1_plus(model) * model.U
beta_2(model::BaroclinicModel) = model.beta - S2_minus(model) * model.U 

function evolve_zeta_top(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, timestep::Int)
    f(zeta::Matrix{Float64}, psi::Matrix{Float64}) = begin
        model.visc*laplace_5p(update_ghost_cells(laplace_5p(psi, model.dx)), model.dx) # Apply ghost cells between laplacian applications.
        - J(model.dx, psi, zeta)
        - beta_1(model)*cd(psi, model.dx)
        - model.U*cd(zeta, model.dx)
    end
    
    z = 1

    if timestep == 1 || timestep == 2    
        # Euler's methed for the first and second step.
        f1 = f(zeta[:,:,z,1], psi[:,:,z,1])
        new_zeta = zeta[:,:,z,1] + model.dt * f1
    else
        # AB3 for subsequent steps. 
        f1 = f(zeta[:,:,z,1], psi[:,:,z,1])
        f2 = f(zeta[:,:,z,2], psi[:,:,z,2])
        f3 = f(zeta[:,:,z,3], psi[:,:,z,3])
        new_zeta = zeta[:,:,z,1] + (model.dt * ((23/12)*f1 - (16/12)*f2 + (5/12)*f3))
    end

    new_zeta = update_ghost_cells(new_zeta)
    zeta = store_new_state(zeta, new_zeta, z)

    return zeta
end

function evolve_zeta_bottom(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, timestep::Int)
    f(zeta::Matrix{Float64}, psi::Matrix{Float64}) = begin
        model.visc*laplace_5p(update_ghost_cells(laplace_5p(psi, model.dx)), model.dx) # Apply ghost cells between laplacian applications.
        - J(model.dx, psi, zeta)
        - beta_2(model)*cd(psi, model.dx)
        - model.r*laplace_5p(psi, model.dx) # Bottom friction
    end
    
    z = 2

    if timestep == 1 || timestep == 2    
        # Euler's methed for the first and second step.
        f1 = f(zeta[:,:,z,1], psi[:,:,z,1])
        new_zeta = zeta[:,:,z,1] + model.dt * f1
    else
        # AB3 for subsequent steps. 
        f1 = f(zeta[:,:,z,1], psi[:,:,z,1])
        f2 = f(zeta[:,:,z,2], psi[:,:,z,2])
        f3 = f(zeta[:,:,z,3], psi[:,:,z,3])
        new_zeta = zeta[:,:,z,1] + (model.dt * ((23/12)*f1 - (16/12)*f2 + (5/12)*f3))
    end

    new_zeta = update_ghost_cells(new_zeta)
    zeta = store_new_state(zeta, new_zeta, z)

    return zeta
end

function evolve_psi(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4})
    # Baroclinic projection to get zeta tilde and psi tilde.
    P_inv = P_inv_matrix(model.H_1, model.H_1)
    zeta_tilde = zeros(Float64, model.M+2, model.P+2, 2, 3)
    psi_tilde = zeros(Float64, model.M+2, model.P+2, 2, 3)
    
    for i in 1:2
        zeta_tilde[:,:,i,1] = P_inv[i,1]*zeta[:,:,1,1] + P_inv[i,2]*zeta[:,:,2,1]
        psi_tilde[:,:,i,1] = P_inv[i,1]*psi[:,:,1,1] + P_inv[i,2]*psi[:,:,2,1]
    end
    
    # 1. Solve the Poisson problem for a.
    f_1 = zeta_tilde[:,:,1,1]
    a = sp_solve_modified_helmholtz(0.0, model.M, model.P, f_1, model.dx)
    a_re = reshape(a, (model.M+2, model.P+2))
    zeta_tilde = store_new_state(zeta_tilde, a_re, 1)
    
    # 2. Solve the modified Helmholtz problem for b
    lambda = -1 / (model.R_d^2) # Non zero eig, Tr(S).
    f_2 = zeta_tilde[:,:,2,1]
    b = sp_solve_modified_helmholtz(lambda, model.M, model.P, f_2, model.dx)
    b_re = reshape(b, (model.M+2, model.P+2))
    zeta_tilde = store_new_state(zeta_tilde, b_re, 1)

    # Baroclinic projection to get back to zeta and psi.
    P = P_matrix(model.H_1, model.H_1)
    
    for i in 1:2
        # zeta_tilde[:,:,i] = P[i,1]*zeta[:,:,1,1] + P_inv[i,2]*zeta[:,:,2,1]
        new_psi = P[i,1]*psi_tilde[:,:,1,1] + P[i,2]*psi_tilde[:,:,2,1]
        new_psi = update_ghost_cells(new_psi)
        psi = store_new_state(psi, new_psi, i)
    end
 
    return psi
end
