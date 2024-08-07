include("model.jl")
include("schemes/helmholtz.jl")
include("schemes/boundary_conditions.jl")

const MINUTES = 60
const DAY = 60*60*24
const KM = 1000.0
const YEAR = 60*60*24*365

function create_metadata(model::BaroclinicModel)
    sample_interval = 1.0*DAY
    sample_timestep = floor(Int, sample_interval / model.dt)

    metadata = Dict("dt" => model.dt,
        "T" => model.T,
        "sample_interval" => sample_interval,
        "sample_timestep" => sample_timestep
    )

    return metadata
end

function run_model(model::BaroclinicModel, file_name::String)
    zeta, psi = initialise_model(model)
    save_results = true

    if save_results
        metadata = create_metadata(model)
        
        # Create the file and write the initial conditions.
        f = jldopen(file_name, "w") do file
            write(file, "zeta_0", zeta[:,:,:,1])
            write(file, "psi_0", psi[:,:,:,1])
            write(file, "metadata", metadata)
        end
    end 

    # charney stern
    println("Parameters:")
    println("(f_0^2 / N^2): ", ratio_term(model))
    println("S1 = ", S1_plus(model))
    println("S2 = ", S2_minus(model))
    println("Beta_1 = ", beta_1(model))
    println("Beta_2 = ", beta_2(model))
    println("M = $M")
    println("P = $P")
    println("dt = $dt")
    println("T = $T")
    
    total_steps = floor(Int, T / dt)
    println("Total steps = $total_steps")
    println("")

    sample_interval = 1.0*DAY
    sample_timestep = floor(Int, sample_interval / dt)

    @time "Time to init Poisson system" poisson_linsolve = get_poisson_linsolve_A(model.M, model.P, model.dx)
    @time "Time to init modified Helmholtz system" helmholtz_linsolve = get_helmholtz_linsolve_A(model.M, model.P, model.dx, S_eig(model))

    println("Starting timeloop")

    for (timestep, time) in enumerate(1:dt:T) 
        evolve_zeta!(model, zeta, psi, timestep)
        evolve_psi!(model, zeta, psi, poisson_linsolve, helmholtz_linsolve)

        if timestep % floor(total_steps / 25) == 0
            percent_complete = round(100timestep / total_steps)
            println("Progress: $percent_complete %")
        end

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
    H_1 = 1.0*KM
    H_2 = 2.0*KM
    beta = 2*10^-11
    Lx = 4000.0*KM # 4000 km
    Ly = 2000.0*KM # 2000 km
    dt = 15.0*MINUTES # 30 minutes TODO: This needs to be reduced I think for convergence.
    T = 0.5*YEAR  # Expect to wait 90 days before seeing things.
    U = 2.0 # Forcing term of top level.
    M = 8
    dx = Lx / M
    P = Int(Ly / dx)
    visc = 100.0 # Viscosity, 100m^2s^-1
    r = 10^-7 # bottom friction scaler.
    R_d = 40.0*KM # Deformation radious, ~40km. Using 60km for better numerics.
    
    model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d)

    simulation_name = "test25"

    run_model(model, simulation_name)
end

# @time "Total runtime:" main()
