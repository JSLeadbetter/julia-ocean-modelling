using LinearAlgebra

include("schemes/arakawa.jl")
include("schemes/laplacian.jl")
include("schemes/boundary_conditions.jl")

const MINUTES = 60
const DAY = 60*60*24
const KM = 1000.0
const YEAR = 60*60*24*365

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
    initial_kick::Float64
end

"""Outer constructor with default values."""
BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d, initial_kick) = BaroclinicModel(
    H_1, H_2, H_1+H_2, beta, Lx, Ly, RectangularDomain(0, Lx, 0, Ly), dt, T, U, M, P, dx, visc, r, R_d, initial_kick) 

"""Initialise the model with a small random psi and then calculate zeta directly."""
function initialise_model(model::BaroclinicModel)
    @assert sign(beta_1(model)) == -sign(beta_2(model))
    
    # Initialise psi with random scaled noise.
    psi_1 = model.initial_kick * model.U * model.Ly * rand(Float64, (model.M+2, model.P+2))
    psi_2 = model.initial_kick * model.U * model.Ly * rand(Float64, (model.M+2, model.P+2)) 

    update_doubly_periodic_bc!(psi_1)
    update_doubly_periodic_bc!(psi_2)    

    zeta_1 = laplace_5p(psi_1, model.dx) + S1_plus(model) * (psi_2 - psi_1)
    zeta_2 = laplace_5p(psi_2, model.dx) + S2_minus(model) * (psi_1 - psi_2)

    update_doubly_periodic_bc!(zeta_1)
    update_doubly_periodic_bc!(zeta_2)

    zeta = zeros(model.M+2, model.P+2, 2, 3)
    psi = zeros(model.M+2, model.P+2, 2, 3)

    psi[:,:,1,1] = copy(psi_1)
    psi[:,:,2,1] = copy(psi_2)
    zeta[:,:,1,1] = copy(zeta_1)
    zeta[:,:,2,1] = copy(zeta_2)
    
    return zeta, psi
end

"""
Centred-difference scheme for matrices in the x-direction.
Doubly periodic boundary conditions.
"""
function cd(u::Matrix{Float64}, dx::Float64)
    M, P = size(u)
    u_out = zeros(M, P)

    @inbounds for j in 2:P-1
        @inbounds for i in 2:M-1
            u_out[i, j] = 0.5dx^-1*(u[i+1, j] - u[i-1, j])
        end
    end

    update_doubly_periodic_bc!(u_out)
    return u_out
end

"The matrix of eigenvectors of the S matrix."
function P_matrix(H_1::Float64, H_2::Float64)
    P = ones(Float64, 2, 2)
    P[1, 2] = -H_2 / H_1
    return P
end

"The inverse of the matrix of eigenvectors of the P matrix."
function P_inv_matrix(model::BaroclinicModel)
    P = zeros(2, 2)
    a = S1_plus(model)
    b = S2_minus(model)
    P[1, 1] = b
    P[1, 2] = a
    P[2, 1] = -b
    P[2, 2] = b
    return (1 / (a+b)) * P
end

"""Shift the stored states forward by one and insert the latest one at the start."""
function store_new_state!(arr::Array{Float64, 4}, new_state::Matrix{Float64}, z::Int)
    arr[:,:,z,3] = arr[:,:,z,2]
    arr[:,:,z,2] = arr[:,:,z,1]
    arr[:,:,z,1] = new_state
end

"""(f_0/N_0)^2"""
function ratio_term(model::BaroclinicModel)
    return 0.5*(model.H_1 + model.H_2) / ((model.R_d^2) * ((1/model.H_1) + (1/model.H_2)))
end

S1_plus(model::BaroclinicModel) = (2 * ratio_term(model)) / (model.H_1 * (model.H_1 + model.H_2))
S2_minus(model::BaroclinicModel) = (2 * ratio_term(model)) / (model.H_2 * (model.H_1 + model.H_2))

# Modified beta functions.
beta_1(model::BaroclinicModel) = model.beta + (S1_plus(model) * model.U)
beta_2(model::BaroclinicModel) = model.beta - (S2_minus(model) * model.U) 

# Non-zero eigenvalue of the S matrix.
S_eig(model::BaroclinicModel) = -1 / model.R_d^2

function eulers_method(model::BaroclinicModel, f::Function, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, z::Int, f_store::Array{Float64, 4})
    f1 = f(model, zeta[:,:,z,1], psi[:,:,z,1])
    store_new_state!(f_store, f1, z)
    return zeta[:,:,z,1] + (model.dt .* f1)
end

function AB3(model::BaroclinicModel, f::Function, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, z::Int, f_store::Array{Float64, 4})
    f1 = f(model, zeta[:,:,z,1], psi[:,:,z,1])
    store_new_state!(f_store, f1, z) 
    f2 = view(f_store, :, :, z, 2)
    f3 = view(f_store, :, :, z, 3)
    update = (model.dt .* ((23/12).*f1 - (16/12).*f2 + (5/12).*f3))    
    return zeta[:,:,z,1] .+ update
end

"""RHS function used for evolving the top zeta layer."""
function zeta_f1(model::BaroclinicModel, zeta::Matrix{Float64}, psi::Matrix{Float64})
    v_term = model.visc*laplace_5p(laplace_5p(psi, model.dx), model.dx)
    J_term = J(model.dx, zeta, psi)
    beta_term = beta_1(model)*cd(psi, model.dx)
    U_term = model.U*cd(zeta, model.dx)    
    return v_term - J_term - beta_term - U_term 
end

function zeta_f2(model::BaroclinicModel, zeta::Matrix{Float64}, psi::Matrix{Float64})
    v_term = model.visc*laplace_5p(laplace_5p(psi, model.dx), model.dx)
    J_term = J(model.dx, zeta, psi)
    beta_term = beta_2(model)*cd(psi, model.dx)
    r_term = model.r*laplace_5p(psi, model.dx)
    return v_term - J_term - beta_term - r_term
end

function evolve_zeta!(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, timestep::Int, f_store::Array{Float64, 4})
    evolve_zeta_layer!(model, zeta, psi, timestep, 1, zeta_f1, f_store)
    evolve_zeta_layer!(model, zeta, psi, timestep, 2, zeta_f2, f_store)
end

function evolve_zeta_layer!(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, timestep::Int, layer::Int, f::Function, f_store::Array{Float64, 4})
    if timestep == 1 || timestep == 2    
        # Euler's methed for the first and second step.
        new_zeta = eulers_method(model, f, zeta, psi, layer, f_store)
        update_doubly_periodic_bc!(new_zeta)
        store_new_state!(zeta, new_zeta, layer)
    else
        # AB3 for subsequent steps. 
        new_zeta = AB3(model, f, zeta, psi, layer, f_store)
        update_doubly_periodic_bc!(new_zeta)
        store_new_state!(zeta, new_zeta, layer)
    end
end

function evolve_psi!(model::BaroclinicModel, zeta::Array{Float64, 4}, psi::Array{Float64, 4}, poisson_cholesky::SparseArrays.CHOLMOD.Factor, helmholtz_cholesky::SparseArrays.CHOLMOD.Factor)
    P = P_matrix(model.H_1, model.H_1)
    P_inv = P_inv_matrix(model)
    zeta_tilde = zeros(Float64, model.M+2, model.P+2, 2, 3)
    psi_tilde = zeros(Float64, model.M+2, model.P+2, 2, 3)

    # Baroclinic projection to get zeta tilde and psi tilde.
    for i in 1:2
        zeta_tilde[:,:,i,1] = P_inv[i,1]*zeta[:,:,1,1] + P_inv[i,2]*zeta[:,:,2,1]
        psi_tilde[:,:,i,1] = P_inv[i,1]*psi[:,:,1,1] + P_inv[i,2]*psi[:,:,2,1]
    end

    # Solve the Poisson problem for the top layer.
    b = -vec(copy(zeta_tilde[2:end-1,2:end-1,1,1])); b[1] = 0
    u = reshape(poisson_cholesky \ b, (model.M, model.P)) 
    new_psi_tilde_1 = add_doubly_periodic_boundaries(u) 
    
    # Solve the modified Helmholtz problem for the bottom layer.
    b = -vec(copy(zeta_tilde[2:end-1,2:end-1,2,1])); b[1] = 0
    u = reshape(helmholtz_cholesky \ b, (model.M, model.P)) 
    new_psi_tilde_2 = add_doubly_periodic_boundaries(u) 

    # Baroclinic projection to get back to zeta and psi.
    for i in 1:2
        new_psi = P[i,1]*new_psi_tilde_1 + P[i,2]*new_psi_tilde_2
        update_doubly_periodic_bc!(new_psi)
        store_new_state!(psi, new_psi, i)
    end
end
