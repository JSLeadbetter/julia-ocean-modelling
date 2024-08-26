"""Plot a single frame of a simulation as a heatmap"""

include("animation.jl")

const MINUTES = 60
const KM = 1000.0
const YEAR = 60*60*24*365

data_file_name = ARGS[1]

# 14.44 = Roughly similar to 11pt in Latex.
fontsize = 28

metadata = get_metadata(data_file_name)
sample_timestep = metadata["sample_timestep"]
dt = metadata["dt"]
println("dt: ", dt)
println("sample timestep: ", sample_timestep)

if haskey(metadata, "total_steps")
    total_steps = metadata["total_steps"]
else
    T = floor(Int, metadata["T"])
    total_steps = floor(Int, T / dt)
end

# For data_file_name = "../../data/double_res_of_first_correct_run.jld":
# 1000 - before breakdown, nice vertical lines.
# 1200 - just after breakdown, good show of initial mixing.
# 3000 - Good jets.

# second_fully_correct
# 350 good vertical stripes
# 500
# 1000
# 2000

sample_no = ARGS[2]

timestep = parse(Int64, sample_no) * sample_timestep
println("timestep: ", timestep)

real_time = timestep * dt
real_time_in_days = real_time / DAY
println("real_time: ", real_time)
println("real_time_in_days: ", real_time_in_days)

zeta = load_matrix(timestep, data_file_name, "zeta")
psi = load_matrix(timestep, data_file_name, "psi")


Lx = 4000.0KM 
Ly = 2000.0KM 
M = 512
dx = Lx / M
P = Int(Ly / dx)

xs = range(-dx, Lx, length=512+2)
ys = range(-dx, Ly, length=256+2)

println(xs)
println(ys)

zeta_top = zeta[:,:,1]

fig = Figure(size = (1200, 800), figure_padding = 10)
ga = fig[1, 1] = GridLayout()

ax = Axis(ga[1, 1], xlabel="x (m)", ylabel="y (m)")

ax.xlabelsize = fontsize
ax.xticklabelsize = fontsize
ax.yticklabelsize = fontsize
ax.ylabelsize = fontsize

hm = heatmap!(ax, xs, ys, zeta_top)
Colorbar(ga[1, 2], hm, ticklabelsize=fontsize)

save("plots/better/zeta_top_$sample_no.png", fig)

# wait(display(fig))