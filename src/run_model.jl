using ProgressBars
using JLD

include("model.jl")
include("schemes/helmholtz.jl")
include("schemes/boundary_conditions.jl")

function create_metadata(model::BaroclinicModel)
    sample_interval = 1.0*DAY
    sample_timestep = floor(Int, sample_interval / model.dt)
    total_steps = floor(Int, model.T / model.dt)

    metadata = Dict(
        "dt" => model.dt,
        "T" => model.T,
        "sample_interval" => sample_interval,
        "sample_timestep" => sample_timestep,
        "total_steps" => total_steps,
    )

    return metadata
end

function log_model_params(model::BaroclinicModel)
    total_steps = floor(Int, model.T / model.dt)
    println("Parameters:")
    println("Lx = ", model.Lx)
    println("Ly = ", model.Ly)
    println("(f_0^2 / N^2): ", ratio_term(model))
    println("S1 = ", S1_plus(model))
    println("S2 = ", S2_minus(model))
    println("Beta_1 = ", beta_1(model))
    println("Beta_2 = ", beta_2(model))
    println("M = ", model.M)
    println("P = ", model.P)
    println("dt = ", model.dt)
    println("T = ", model.T)
    println("U = ", model.U)
    println("Initial kick = ", model.initial_kick)
    println("Total steps = ", total_steps, "\n")
end

function run_model(model::BaroclinicModel, file_name::String, save_results::Bool)
    zeta, psi = initialise_model(model)

    if save_results
        metadata = create_metadata(model)
        
        # Create the file and write the initial conditions.
        f = jldopen(file_name, "w") do file
            write(file, "zeta_0", zeta[:,:,:,1])
            write(file, "psi_0", psi[:,:,:,1])
            write(file, "metadata", metadata)
        end
    end 

    log_model_params(model)

    sample_interval = 1.0*DAY
    sample_timestep = 2*floor(Int, sample_interval / model.dt)

    @time "Time to Cholesky factorise Poisson system" poisson_chol_fact = get_poisson_cholesky(model.M, model.P, model.dx)
    @time "Time to Cholesky factorise modified Helmholtz system" helmholtz_chol_fact = get_helmholtz_cholesky(model.M, model.P, model.dx, S_eig(model))

    total_steps = floor(Int, model.T / model.dt)

    println("Running simulation... \n")

    # for timestep in 1:total_steps
    for timestep in ProgressBar(1:total_steps)
        evolve_zeta!(model, zeta, psi, timestep)
        evolve_psi!(model, zeta, psi, poisson_chol_fact, helmholtz_chol_fact)

        if timestep % sample_timestep == 0 && save_results
            f = jldopen(file_name, "r+") do file
                write(file, "zeta_$timestep", zeta[:,:,:,1])
                write(file, "psi_$timestep", psi[:,:,:,1])
            end
        end
    end

    return zeta, psi
end

function main()
    H_1 = 0.5*KM
    H_2 = 2.0*KM
    beta = 2*10^-11
    Lx = 4000.0*KM # 4000 km
    Ly = 4000.0*KM # 2000 km
    dt = 15.0*MINUTES # 30 minutes
    T = 10.0YEAR  # Expect to wait 90 days before seeing things.
    U = 0.1 # Forcing term of top level.
    M = P = 256
    dx = Lx / M
    # P = Int(Ly / dx)
    visc = 100.0 # Viscosity, 100m^2s^-1
    r = 10^-8 # bottom friction scaler.
    R_d = 40.0*KM # Deformation radius, ~40km. Using 60km for better numerics.
    initial_kick = 1e-6

    model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d, initial_kick)

    sim_name = "thinner_upper"
    data_file_name = "data/$sim_name.jld"

    println("Saving simulation results to: ", data_file_name)

    run_model(model, data_file_name, true)
end

# @time "\n Total runtime:" main()
