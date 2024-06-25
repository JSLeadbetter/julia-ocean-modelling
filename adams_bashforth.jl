using LinearAlgebra
using Plots

# Adams-Bashforth in the degenerate s = 1 case.
function eulers_method(h::Float64, f::Function, y_0::Float64, M::Float64)
    N = ceil(Int64, M / h)
    u = zeros(Float64, N+1)
	t = range(0, M, step=h)

	# Set the initial condition
    u[1] = y_0

    for i in 1:N-1
        u[i+1] = u[i] + h * f(t[i], u[i])
    end

    return u, t
end

function adams_bashforth_2nd_order(h::Float64, f::Function, y_0::Float64, M::Float64)
    N = ceil(Int64, M / h)
    u = zeros(Float64, N+1)
    t = range(0, M, step=h)

    u[1] = y_0

	# Calculate y_1 using Euler's method first.
	u_em, _ = eulers_method(h, f, y_0, h)
	u[2] = u_em[2]

    for i in 1:N-2
        u[i+2] = u[i+1] + ((3/2)*h*f(t[i+1], u[i+1])) - ((1/2)*h*f(t[i], u[i]))
    end

    return u, t
end
