include("model.jl")

function run_model_no_output(model::BaroclinicModel)
    zeta, psi = initialise_model(model)
    poisson_chol_fact = get_poisson_cholesky(model.M, model.P, model.dx)
    helmholtz_chol_fact = get_helmholtz_cholesky(model.M, model.P, model.dx, S_eig(model))
    total_steps = floor(Int, model.T / model.dt)

    for timestep in 1:total_steps
        evolve_zeta!(model, zeta, psi, timestep)
        evolve_psi!(model, zeta, psi, poisson_chol_fact, helmholtz_chol_fact)
    end

    return zeta, psi
end