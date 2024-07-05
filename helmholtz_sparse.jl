using LinearAlgebra
using SparseArrays
using IterativeSolvers
using LinearSolve

include("toolbox.jl")

function b_mp(x::Float64, y::Float64)
    term_1 = 2 * ((x + 1)^2) * (x - 1)
    term_2 = (6 * x + 2) * (y + 1) * (y - 1)
    term_3 = 3 * ((x + 1)^2) * (x - 1) * (y + 1) * (y - 1)
    term_4 = (6 - 10 * (pi^2)) * sin(2pi * x) * sin(pi * y)
    return sum([term_1, term_2, term_3, term_4])
end

# A matrix B which is used as blocks inside the matrix A.
function construct_spB(M::Int, alpha::Float64, dx::Float64)
    diag_vals = diag_vals = (-4 + (alpha * dx^2)) * ones(M-1)
    B = spdiagm(0 => diag_vals, 1 => ones(M-2), -1 => ones(M-2))
	return B
end

# Our matrix A in our linear system Ax = b.
function construct_spA(M::Int, alpha::Float64, dx::Float64)
	B = construct_spB(M, alpha, dx)
    A = kron(speye(M-1), B) + kron(spdiagm(1 => ones(M-2), -1 => ones(M-2)), speye(M-1))
	return A * 1 / (dx)^2
end

function get_dx(domain::Vector{Float64}, M::Int64)
    x1 = domain[1]
    x2 = domain[2]
    x_length = x2 - x1
    return x_length / M
end

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

function compute_u(x::Float64, y::Float64)
    return ((x + 1)^2) * (x - 1) * (y + 1) * (y - 1) + 2sin(2*pi*x) * sin(pi*y)
end

function get_xs_ys(domain::Vector{Float64}, M::Int64)
    x1 = domain[1]
    x2 = domain[2]
    y1 = domain[3]
    y2 = domain[4]

    dx = get_dx(domain, M)
    
    xs = range(x1+dx, x2-dx, length=M-1)
    ys = range(y1+dx, y2-dx, length=M-1)
    return xs, ys
end

# alpha = 3.0
# domain = [-1.0, 1.0, -1.0, 1.0]
# M_list = [16, 32, 64, 128, 256, 512]

# for M in M_list
#     u = sp_solve_modified_helmholtz(alpha, M, b_mp, domain)
#     xs, ys = get_xs_ys(domain, M)
#     u_true = vec(inflate(compute_u, xs, ys))
#     println(norm(u - u_true))
# end
