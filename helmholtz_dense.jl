using LinearAlgebra
using Plots

# Helper functions.
eye(n::Int) = Matrix{Float64}(I, n, n)
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

function b_mp(x::Float64, y::Float64)
    term_1 = 2 * ((x + 1)^2) * (x - 1)
    term_2 = (6 * x + 2) * (y + 1) * (y - 1)
    term_3 = 3 * ((x + 1)^2) * (x - 1) * (y + 1) * (y - 1)
    term_4 = (6 - 10 * (pi^2)) * sin(2pi * x) * sin(pi * y)
    return sum([term_1, term_2, term_3, term_4])
end

function construct_B(M::Int, alpha::Float64, dx::Float64)
	diag_vals = (-4 + (alpha * dx^2)) * ones(M-1) 
	B = diagm(0 => diag_vals, 1 => ones(M-2), -1 => ones(M-2))
	return B
end

function construct_A(M::Int, alpha::Float64, dx::Float64)
	B = construct_B(M, alpha, dx)
	A = kron(eye(M-1), B) + kron(diagm(1 => ones(M-2), -1 => ones(M-2)), eye(M-1))
	return A * 1 / (dx)^2
end

function compute_u(x::Float64, y::Float64)
    return ((x + 1)^2) * (x - 1) * (y + 1) * (y - 1) + 2sin(2*pi*x) * sin(pi*y)
end

function solve_modified_helmholtz(alpha::Float64, M::Int64, b_rhs_func::Function, x1::Float64, x2::Float64, y1::Float64, y2::Float64)
    # Calculate the stepsize using the number of discrete points in our grid.
    dx = 2 / M

    # Construct our matrix A.
    A = construct_A(M, alpha, dx)

    # Define the points on our domain.
    xs = range(x1+dx, x2-dx, length=M-1)
    ys = range(y1+dx, y2-dx, length=M-1)

    # Construct the RHS of our equation.
    b = vec(inflate(b_rhs_func, xs, ys))

    # Solve the system.
    return A \ b
end

# Discrete points in our grid.
M = 64

# Modified helmholtz parameter.
alpha = 3.0

# Solve the system.
# u = solve_modified_helmholtz(alpha, M, b_mp)

# xs = range(x_b1+dx, x_b2-dx, length=M-1)
# ys = range(y_b1+dx, y_b2-dx, length=M-1)

# # Reshape the solution for plotting.
# u_re = reshape(u, M-1, M-1)

# # Compute the true solution to the equation.
# u_true = vec(inflate(compute_u, xs, ys))
# # and reshape for plotting.
# u_true_re = reshape(u_true, M-1, M-1)

# # Compute the 2-norm between our numerical approximation and the true solution.
# norm(u - u_true)
