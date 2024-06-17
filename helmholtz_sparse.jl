using LinearAlgebra
using SparseArrays

speye(n::Int) = spdiagm(ones(n))
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

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
	B = construct_B(M, alpha, dx)
    A = kron(speye(M-1), B) + kron(spdiagm(1 => ones(M-2), -1 => ones(M-2)), speye(M-1))
	return A * 1 / (dx)^2
end

function sp_solve_modified_helmholtz(alpha::Float64, M::Int64, b_rhs_func::Function, x1::Float64, x2::Float64, y1::Float64, y2::Float64)
    # Stepsize.
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
M = 4

# Modified helmholtz parameter.
alpha = 3.0

# Stepsize.
dx = 2 / M

# solve_modified_helmholtz(alpha, M, b_mp)