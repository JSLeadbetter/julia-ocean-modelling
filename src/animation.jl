using GLMakie
using JLD

const DAY = 60*60*24

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

function create_mp4(data_file_name::String, animation_file_name::String, fps::Int)
    metadata = get_metadata(data_file_name)
    sample_timestep = 2*metadata["sample_timestep"]
    dt = metadata["dt"]

    if haskey(metadata, "total_steps")
        total_steps = metadata["total_steps"]
    else
        T = floor(Int, metadata["T"])
        total_steps = floor(Int, T / dt)
    end

    shift_amounts = (0, 0)

    zeta = load_matrix(0, data_file_name, "zeta")
    psi = load_matrix(0, data_file_name, "psi")

    zeta_top_shifted = circshift(zeta[:,:,1], shift_amounts)
    zeta_bottom_shifted = circshift(zeta[:,:,2], shift_amounts)

    zeta_top_plot = Observable(zeta_top_shifted)
    zeta_bottom_plot = Observable(zeta_bottom_shifted)

    psi_top_shifted = circshift(psi[:,:,1], shift_amounts)
    psi_bottom_shifted = circshift(psi[:,:,2], shift_amounts)

    psi_top_plot = Observable(psi_top_shifted)
    psi_bottom_plot = Observable(psi_bottom_shifted)

    plot_time = Observable(0.0)

    fig = Figure(size = (1200, 800), figure_padding = 80)
    ga = fig[1, 1] = GridLayout()
    
    Label(fig[0, :], lift(t -> "Current Time (Days) = $(round(t, digits=1))", plot_time), 
        fontsize = 24, tellwidth = false)

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

    # display(fig)

    sleep_time = 1 / fps

    record(fig, animation_file_name, 0:sample_timestep:total_steps, framerate = fps) do timestep
        zeta = load_matrix(Int(timestep), data_file_name, "zeta")
        psi = load_matrix(Int(timestep), data_file_name, "psi")

        zeta_top_shifted = circshift(zeta[:,:,1], shift_amounts)
        zeta_bottom_shifted = circshift(zeta[:,:,2], shift_amounts) 

        zeta_top_plot[] = zeta_top_shifted
        zeta_bottom_plot[] = zeta_bottom_shifted

        psi_top_shifted = circshift(psi[:,:,1], shift_amounts)
        psi_bottom_shifted = circshift(psi[:,:,2], shift_amounts) 

        psi_top_plot[] = psi_top_shifted
        psi_bottom_plot[] = psi_bottom_shifted

        plot_time[] = timestep * dt / DAY

        # display(fig)
        # sleep(sleep_time)
    end

    # Will keep the plot open until it is closed, otherwise it will close at the end of the animation.
    # wait(display(fig))
end

function show_animation(file_name::String, fps::Int)
    metadata = get_metadata(file_name)
    println(metadata)

    sample_timestep = 2*metadata["sample_timestep"]
    dt = metadata["dt"]

    if haskey(metadata, "total_steps")
        total_steps = metadata["total_steps"]
    else
        T = floor(Int, metadata["T"])
        total_steps = floor(Int, T / dt)
    end

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

    plot_time = Observable(0.0)

    fig = Figure(size = (1200, 800), figure_padding = 80)
    ga = fig[1, 1] = GridLayout()
    
    Label(fig[0, :], lift(t -> "Current Time (Days) = $(round(t, digits=1))", plot_time), 
        fontsize = 24, tellwidth = false)

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

        zeta_top_shifted = circshift(zeta[:,:,1], shift_amounts)
        zeta_bottom_shifted = circshift(zeta[:,:,2], shift_amounts) 

        zeta_top_plot[] = zeta_top_shifted
        zeta_bottom_plot[] = zeta_bottom_shifted

        psi_top_shifted = circshift(psi[:,:,1], shift_amounts)
        psi_bottom_shifted = circshift(psi[:,:,2], shift_amounts) 

        psi_top_plot[] = psi_top_shifted
        psi_bottom_plot[] = psi_bottom_shifted

        plot_time[] = timestep * dt / DAY

        display(fig)
        sleep(sleep_time)
    end

    # Will keep the plot open until it is closed, otherwise it will close at the end of the animation.
    wait(display(fig))
end

begin
    data_file = ARGS[1]
    fps = 30
    
    # Display animation
    println("Showing animation for datafile: ", data_file)
    show_animation(data_file, 30)
    
    # Create .mp4 file.
    # animation_file_name = "its_working.mp4"
    # println("Creating .mp4 for datafile: ", data_file)
    # create_mp4(data_file, animation_file_name, 30) 
end
