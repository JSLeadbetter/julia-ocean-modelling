using BenchmarkTools, CSV, DataFrames

include("../run_model_no_output.jl")

# Standard parameters.
H_1 = 1.0*KM
H_2 = 2.0*KM
beta = 2*10^-11
Lx = 4000.0*KM # 4000 km
Ly = 4000.0*KM # 2000 km
U = 0.1 # Forcing term of top level.
visc = 100.0 # Viscosity, 100m^2s^-1
r = 10^-7 # bottom friction scaler.
R_d = 40.0*KM # Deformation radious, ~40km. Using 60km for better numerics.
initial_kick = 1e-6

dt = 30.0MINUTES # 30 minutes
T = 30.0DAY  # Expect to wait 90 days before seeing things.
M_list = [8, 16, 32, 64, 128]
sample_size = 50
evals_size = 50
seconds = 120 # Seconds to wait before stopping benchmarking.

total_times = zeros(size(M_list)[1])
zeta_times = zeros(size(M_list)[1])
psi_times = zeros(size(M_list)[1])
helm_times = zeros(size(M_list)[1])
poisson_times = zeros(size(M_list)[1]) 

for (i, M) in enumerate(M_list)
    P = M
    dx = Lx / M
    model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, dt, U, M, P, dx, visc, r, R_d, initial_kick)
    println("Benchmarking for M = $M")
    
    @time "total benchmarktime" total_times[i] = @belapsed run_model_no_output($model) samples=sample_size seconds=seconds evals=evals_size
    
    f_store = zeros(model.M+2, model.P+2, 2, 3)
    helm_chol = get_helmholtz_cholesky(M, P, dx, S_eig(model))
    poisson_chol = get_poisson_cholesky(M, P, dx)
    zeta, psi = initialise_model(model)

    helm_times[i] = @belapsed get_helmholtz_cholesky($M, $P, $dx, S_eig($model))
    poisson_times[i] = @belapsed get_poisson_cholesky($M, $P, $dx)

    # @btime get_helmholtz_cholesky(M, P, dx, S_eig(model))
    @time "psi bench time" psi_times[i] = @belapsed evolve_psi!($model, $zeta, $psi, $poisson_chol, $helm_chol) samples=sample_size seconds=seconds evals=evals_size
    @time "zeta bench time" zeta_times[i] = @belapsed evolve_zeta!($model, $zeta, $psi, 1, $f_store) samples=sample_size seconds=seconds evals=evals_size
end

df = DataFrame("M" => M_list, "total_time" => total_times, "psi_time" => psi_times, "zeta_time" => zeta_times, "helmholtz_time" => helm_times, "poisson_times" => poisson_times)
CSV.write("julia_parts_benchmark.csv", df)

# M = M_list[1]
# P = M[1]
# dx = Lx / M
# model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d, initial_kick)
# display(@benchmark run_model_no_output(model) samples=50)

# one_step_model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, dt, U, M, P, dx, visc, r, R_d, initial_kick)

# println("Full model:")
# @btime run_model_no_output(model)

# println("One step model:")
# @btime run_model_no_output(one_step_model)

# helm_chol = get_helmholtz_cholesky(M, P, dx, S_eig(model))
# poisson_chol = get_poisson_cholesky(M, P, dx)
# println("Helmholtz chol:")
# @btime get_helmholtz_cholesky(M, P, dx, S_eig(model))

# zeta, psi = initialise_model(model)

# println("Evolve psi:")
# @btime evolve_psi!(model, zeta, psi, poisson_chol, helm_chol)

# println("Evolve zeta:")
# @btime evolve_zeta!(model, zeta, psi, 1)
# zeta_1 = zeta[:,:,1,1]
# psi_1 = psi[:,:,1,1]

# println("Viscosity Laplacian:")
# @btime model.visc*laplace_5p(laplace_5p(psi_1, model.dx), model.dx)

# println("AB3")
# @btime AB3(model, zeta_f1, zeta, psi, 1)

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



# println("Arakawa Jacobian:")
# @btime J(model.dx, zeta_1, psi_1)

# println("Beta 1 Centred Difference:")
# @btime beta_1(model)*cd(psi_1, model.dx)

# println("U Centred Difference:")
# @btime model.U*cd(zeta_1, model.dx)
