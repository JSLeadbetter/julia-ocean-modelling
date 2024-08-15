using BenchmarkTools
using LinearSolve

include("run_model.jl")

# const MINUTES = 60
# const DAY = 60*60*24
# const KM = 1000.0
# const YEAR = 60*60*24*365

H_1 = 1.0*KM
H_2 = 2.0*KM
beta = 2*10^-11
Lx = 4000.0*KM # 4000 km
Ly = 4000.0*KM # 2000 km
dt = 15.0*MINUTES # 30 minutes
T = 1.0*YEAR  # Expect to wait 90 days before seeing things.
U = 0.1 # Forcing term of top level.
M = P = 128
dx = Lx / M
# P = Int(Ly / dx)
visc = 100.0 # Viscosity, 100m^2s^-1
r = 10^-7 # bottom friction scaler.
R_d = 40.0*KM # Deformation radious, ~40km. Using 60km for better numerics.
initial_kick = 1e-6

model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d, initial_kick)
@time run_model(model, "", false)

# poisson_linsolve = get_poisson_linsolve_A(model.M, model.P, model.dx)

# println("Poisson linsolve")
# @btime get_poisson_linsolve_A(model.M, model.P, model.dx)
# # @btime helmholtz_linsolve = get_helmholtz_linsolve_A(model.M, model.P, model.dx, S_eig(model))
# # chol_fac = get_cholesky_factorisation(model.M, model.P, model.dx, 0.0)
# # @btime get_cholesky_factorisation(model.M, model.P, model.dx, S_eig(model))

# A = -construct_spA(M, P, dx, 0.0)

# # Ensure matrix is positive definite by reducing number of unknowns.
# A[:,1] .= 0
# A[1,:] .= 0
# A[1, 1] = 1

# println(typeof(A))

# @assert typeof(A) == SparseMatrixCSC{Float64, Int64}

# println("Cholesky factorisation")
# @btime cholesky(A)
# chol_A = cholesky(A)#LinearSolve.factorize(A)


# zeta, psi = initialise_model(model)

# zeta_1 = zeta[:,:,1,1]
# psi_1 = psi[:,:,1,1]

# P = P_matrix(model.H_1, model.H_1)
# P_inv = P_inv_matrix(model)
# zeta_tilde = zeros(Float64, model.M+2, model.P+2, 2, 3)
# psi_tilde = zeros(Float64, model.M+2, model.P+2, 2, 3)

# # # Baroclinic projection to get zeta tilde and psi tilde.
# for i in 1:2
#     zeta_tilde[:,:,i,1] = P_inv[i,1]*zeta[:,:,1,1] .+ P_inv[i,2]*zeta[:,:,2,1]
#     psi_tilde[:,:,i,1] = P_inv[i,1]*psi[:,:,1,1] .+ P_inv[i,2]*psi[:,:,2,1]
# end

# # # Solve the Poisson problem for the top layer.
# f_1 = copy(zeta_tilde[:,:,1,1])
# update_doubly_periodic_bc!(f_1)

# b = -vec(f_1[2:end-1, 2:end-1])
# b[1] = 0
# poisson_linsolve.b = b

# println("Linsolve solve")
# @btime u = solve(poisson_linsolve).u

# println("Cholesky solve")
# @btime chol_A \ b

# prob = LinearProblem(chol_fac, b)
# @btime solve(prob)
# # u_re = reshape(u, (model.M, model.P))
# # new_psi_tilde_1 = add_doubly_periodic_boundaries(u_re)

# # # Solve the modified Helmholtz problem for the bottom layer.
# f_2 = copy(zeta_tilde[:,:,2,1])
# update_doubly_periodic_bc!(f_2)

# b = -vec(f_2[2:end-1, 2:end-1])
# b[1] = 0
# helmholtz_linsolve.b = b
# @btime u = solve(helmholtz_linsolve).u
# u_re = reshape(u, (model.M, model.P))
# new_psi_tilde_2 = add_doubly_periodic_boundaries(u_re)

# # store_new_state!(psi, new_psi_tilde_1, 1)
# # store_new_state!(psi, new_psi_tilde_2, 2)

# # Baroclinic projection to get back to zeta and psi.
# for i in 1:2
#     new_psi = P[i,1]*new_psi_tilde_1 + P[i,2]*new_psi_tilde_2
#     update_doubly_periodic_bc!(new_psi)
#     store_new_state!(psi, new_psi, i)
# end


# println("Evolve Zeta:")
# @btime evolve_zeta!(model, zeta, psi, 1)

# println("Evolve Psi:")
# @btime evolve_psi!(model, zeta, psi, poisson_linsolve, helmholtz_linsolve)

# println("Viscosity Laplacian:")
# @btime model.visc*laplace_5p(laplace_5p(psi_1, model.dx), model.dx)

# println("Arakawa Jacobian:")
# @btime J(model.dx, zeta_1, psi_1)

# println("Beta 1 Centred Difference:")
# @btime beta_1(model)*cd(psi_1, model.dx)

# println("U Centred Difference:")
# @btime model.U*cd(zeta_1, model.dx)
