### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 63804f65-b530-4b6f-aa96-050c9ca0e494
using LinearAlgebra

# ╔═╡ e6517dd3-18b8-4d7a-adfc-45de3523f964
using SparseArrays

# ╔═╡ 80520e35-8548-4c27-b301-bd9beec40107
using IterativeSolvers

# ╔═╡ de88f177-0ae7-485f-ab00-6696c43af33d
using LinearSolve

# ╔═╡ d43de74b-6084-426f-a431-1a197d659525
using Plots

# ╔═╡ 21f239b1-ab9a-42ca-9515-746da51a7fdc
speye(n::Int) = spdiagm(ones(n))

# ╔═╡ cb9fb5f3-a7bb-4629-86a9-b68bfd9c9f7e
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

# ╔═╡ de6c847b-d3d1-4d19-8fe6-f984b6eee6f2
function b_mp(x::Float64, y::Float64)
    term_1 = 2 * ((x + 1)^2) * (x - 1)
    term_2 = (6 * x + 2) * (y + 1) * (y - 1)
    term_3 = 3 * ((x + 1)^2) * (x - 1) * (y + 1) * (y - 1)
    term_4 = (6 - 10 * (pi^2)) * sin(2pi * x) * sin(pi * y)
    return sum([term_1, term_2, term_3, term_4])
end

# A matrix B which is used as blocks inside the matrix A.

# ╔═╡ 080ae91d-a7d9-46a2-b88b-4d733b30a92d
function construct_spB(M::Int, alpha::Float64, dx::Float64)
    diag_vals = diag_vals = (-4 + (alpha * dx^2)) * ones(M-1)
    B = spdiagm(0 => diag_vals, 1 => ones(M-2), -1 => ones(M-2))
	return B
end

# Our matrix A in our linear system Ax = b.

# ╔═╡ 8e953aee-7eb7-49b8-9fab-60221dba16e6
function construct_spA(M::Int, alpha::Float64, dx::Float64)
	B = construct_spB(M, alpha, dx)
    A = kron(speye(M-1), B) + kron(spdiagm(1 => ones(M-2), -1 => ones(M-2)), speye(M-1))
	return A * 1 / (dx)^2
end

# ╔═╡ 076cb0e0-a87a-4db9-b613-806ec6e4db23
function get_dx(domain::Vector{Float64}, M::Int64)
    x1 = domain[1]
    x2 = domain[2]
    x_length = x2 - x1
    return x_length / M
end

# ╔═╡ db666ad0-113f-4178-942b-844e81a04ad4
function laplace_2d(u::Matrix{Float64}, h::Float64)
    nx, ny = size(u)
    lap_u = zeros(nx, ny)
    
    # Apply the Laplace operator using central differences
    for i in 2:nx-1
        for j in 2:ny-1
            lap_u[i, j] = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / h^2 +
                          (u[i, j+1] - 2u[i, j] + u[i, j-1]) / h^2
        end
    end
    
    return lap_u
end

# ╔═╡ 194994fc-1e00-4e0b-9983-36614e10f0a7
function compute_u(x::Float64, y::Float64)
    return ((x + 1)^2) * (x - 1) * (y + 1) * (y - 1) + 2sin(2*pi*x) * sin(pi*y)
end

# ╔═╡ 0ed9a57b-6cb1-4b08-aaa5-a3a352dc146f
function get_xs_ys(domain::Vector{Float64}, M::Int64)
    x1, x2, y1, y2 = domain

    dx = get_dx(domain, M)
    
    xs = range(x1+dx, x2-dx, length=M-1)
    ys = range(y1+dx, y2-dx, length=M-1)
    return xs, ys
end

# ╔═╡ c8875b9a-d1ed-4571-9e99-ff90f8fa1d91
function sp_solve_modified_helmholtz(alpha::Float64, M::Int64, b_rhs_func::Function, domain::Vector{Float64})
    # Stepsize.
    dx = get_dx(domain, M)

    # Construct our matrix A.
    A = construct_spA(M, alpha, dx)

    # Define the points on our domain.
    xs, ys = get_xs_ys(domain, M)

    # Construct the RHS of our equation.
    b = vec(inflate(b_rhs_func, xs, ys))

    # Solve the system.
    prob = LinearProblem(A, b)
    linsolve = init(prob)
    return solve(linsolve).u
end

# ╔═╡ 3f5d60ba-e87c-4337-99b5-e98e005bedd9
alpha = 3.0

# ╔═╡ 662fc367-354e-4688-aca3-c4cfc91819b3
domain = [-1.0, 1.0, -1.0, 1.0]

# ╔═╡ 89be285a-3cf8-445e-8d5d-dd8515e38562
M = 512

# ╔═╡ 11b7cde7-026a-492d-b643-1ea9194cddb1
begin
	u = sp_solve_modified_helmholtz(alpha, M, b_mp, domain)
	xs, ys = get_xs_ys(domain, M)
	u_true = vec(inflate(compute_u, xs, ys))
	
	# Reshape the solution for plotting.
	u_re = reshape(u, M-1, M-1)
	
	# Compute the true solution to the equation.
	u_true = vec(inflate(compute_u, xs, ys))
	# and reshape for plotting.
	u_true_re = reshape(u_true, M-1, M-1)
	println("Norm: ", norm(u - u_true))
end

# ╔═╡ 16a7e761-a8df-4e15-9c43-e94438a5b93c
begin
	x1, x2, y1, y2 = domain
	plot(xs, ys, u_re, st=:surface, xlabel="x", ylabel="y")
end

# ╔═╡ 700fb47d-8f73-4ca2-979c-1383993a219b
begin
	pts = [(x, y, 0) for x in xs, y in ys]
	plot(vec(pts), surftype=(surface=false, mesh=true), xlim = [-3, 3], ylim=[-3, 3], xlabel="x", ylabel="y")
end

# ╔═╡ Cell order:
# ╠═63804f65-b530-4b6f-aa96-050c9ca0e494
# ╠═e6517dd3-18b8-4d7a-adfc-45de3523f964
# ╠═80520e35-8548-4c27-b301-bd9beec40107
# ╠═de88f177-0ae7-485f-ab00-6696c43af33d
# ╠═d43de74b-6084-426f-a431-1a197d659525
# ╠═21f239b1-ab9a-42ca-9515-746da51a7fdc
# ╠═cb9fb5f3-a7bb-4629-86a9-b68bfd9c9f7e
# ╠═de6c847b-d3d1-4d19-8fe6-f984b6eee6f2
# ╠═080ae91d-a7d9-46a2-b88b-4d733b30a92d
# ╠═8e953aee-7eb7-49b8-9fab-60221dba16e6
# ╠═076cb0e0-a87a-4db9-b613-806ec6e4db23
# ╠═c8875b9a-d1ed-4571-9e99-ff90f8fa1d91
# ╠═db666ad0-113f-4178-942b-844e81a04ad4
# ╠═194994fc-1e00-4e0b-9983-36614e10f0a7
# ╠═0ed9a57b-6cb1-4b08-aaa5-a3a352dc146f
# ╠═3f5d60ba-e87c-4337-99b5-e98e005bedd9
# ╠═662fc367-354e-4688-aca3-c4cfc91819b3
# ╠═89be285a-3cf8-445e-8d5d-dd8515e38562
# ╠═11b7cde7-026a-492d-b643-1ea9194cddb1
# ╠═16a7e761-a8df-4e15-9c43-e94438a5b93c
# ╠═700fb47d-8f73-4ca2-979c-1383993a219b
