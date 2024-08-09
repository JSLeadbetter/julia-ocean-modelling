using GLMakie
using JLD

function load_matrix(timestep::Int, file_name::String, var::String)
    data_name = var * "_$timestep"
    c = jldopen(file_name, "r") do file
        return read(file, data_name)
    end
end

function get_metadata(file_name::String)
    c = jldopen(file_name, "r") do file
        return read(file, "metadata")
    end
end

function show_animation(file_name::String, shift_amounts, fps::Int)
    metadata = get_metadata(file_name)
    println(metadata)

    T = floor(Int, metadata["T"])
    # dt = metadata["dt"]
    sample_timestep = 2*metadata["sample_timestep"]

    total_steps = metadata["total_steps"]

    zeta = load_matrix(0, file_name, "zeta")
    psi = load_matrix(0, file_name, "psi")

    zeta_top_shifted = circshift(zeta[:,:,1], shift_amounts)
    zeta_bottom_shifted = circshift(zeta[:,:,2], shift_amounts)

    zeta_top_plot = Observable(zeta_top_shifted)
    zeta_bottom_plot = Observable(zeta_bottom_shifted)

    psi_top_shifted = circshift(psi[:,:,1], shift_amounts)
    psi_bottom_shifted = circshift(psi[:,:,2], shift_amounts)

    psi_top_plot = Observable(psi_top_shifted)
    psi_bottom_plot = Observable(psi_bottom_shifted)

    # plot_time = Observable(0.0)

    fig = Figure(size = (1200, 800))
    ga = fig[1, 1] = GridLayout()
    
    ax_top_left = Axis(ga[1, 1], title="Zeta Layer 1")
    ax_bottom_left = Axis(ga[2, 1], title="Zeta Layer 2")
    ax_top_right = Axis(ga[1, 3], title="Psi Layer 1")
    ax_bottom_right = Axis(ga[2, 3], title="Psi Layer 2")

    hm1 = heatmap!(ax_top_left, zeta_top_plot)
    hm2 = heatmap!(ax_bottom_left, zeta_bottom_plot)
    hm3 = heatmap!(ax_top_right, psi_top_plot)
    hm4 = heatmap!(ax_bottom_right, psi_bottom_plot)

    Colorbar(ga[1, 2], hm1)
    Colorbar(ga[2, 2], hm2)
    Colorbar(ga[1, 4], hm3)
    Colorbar(ga[2, 4], hm4)

    display(fig)

    sleep_time = 1 / fps

    for timestep in 0:sample_timestep:total_steps
        zeta = load_matrix(Int(timestep), file_name, "zeta")
        psi = load_matrix(Int(timestep), file_name, "psi")

        # println("$file_name, $timestep, zeta top | ", zeta[10,10,1])
        # println("$file_name, $timestep, zeta bottom | ", zeta[10,10,2])
        # println("$file_name, $timestep, psi top | ", psi[10,10,1])
        # println("$file_name, $timestep, psi bottom | ", psi[10,10,2])

        zeta_top_shifted = circshift(zeta[:,:,1], shift_amounts)
        zeta_bottom_shifted = circshift(zeta[:,:,2], shift_amounts) 

        zeta_top_plot[] = zeta_top_shifted
        zeta_bottom_plot[] = zeta_bottom_shifted

        psi_top_shifted = circshift(psi[:,:,1], shift_amounts)
        psi_bottom_shifted = circshift(psi[:,:,2], shift_amounts) 

        psi_top_plot[] = psi_top_shifted
        psi_bottom_plot[] = psi_bottom_shifted

        display(fig)
        sleep(sleep_time)
    end

    # for (timestep, time) in enumerate(0:dt:T)
    #     if timestep % (sample_timestep) == 0
    #         zeta = load_matrix(Int(timestep), file_name, "zeta")
    #         psi = load_matrix(Int(timestep), file_name, "psi")

    #         # println("$file_name, $timestep, zeta top | ", zeta[10,10,1])
    #         # println("$file_name, $timestep, zeta bottom | ", zeta[10,10,2])
    #         # println("$file_name, $timestep, psi top | ", psi[10,10,1])
    #         # println("$file_name, $timestep, psi bottom | ", psi[10,10,2])

    #         zeta_top_shifted = circshift(zeta[:,:,1], shift_amounts)
    #         zeta_bottom_shifted = circshift(zeta[:,:,2], shift_amounts) 

    #         zeta_top_plot[] = zeta_top_shifted
    #         zeta_bottom_plot[] = zeta_bottom_shifted

    #         psi_top_shifted = circshift(psi[:,:,1], shift_amounts)
    #         psi_bottom_shifted = circshift(psi[:,:,2], shift_amounts) 

    #         psi_top_plot[] = psi_top_shifted
    #         psi_bottom_plot[] = psi_bottom_shifted

    #         display(fig)
    #         sleep(sleep_time)
    #     end
    # end

    gui(p)
end

show_animation("data/test_39.jld", (0, 0), 30)
