using LinearAlgebra
using Plots

# Adams-Bashforth in the degenerate s = 1 case.
function eulers_method(h::Float64, f::Function, y_0::Float64, M::Float64)
    N = ceil(Int64, M / h)
    u = zeros(Float64, N+1)
	t = range(0, M, step=h)

	# Set the initial condition
    u[1] = y_0

    for i in 1:N
        u[i+1] = u[i] + h * f(t[i], u[i])
    end

    return u, t
end

function adams_bashforth_2nd_order(h::Float64, f::Function, y_0::Float64, M::Float64)
    N = ceil(Int64, M / h)
    u = zeros(Float64, N+1)
    t = range(0, M, step=h)

    # Set the initial condition.
    u[1] = y_0

	# Calculate y_1 using Euler's method first.
	u_em, _ = eulers_method(h, f, y_0, h)
	u[2] = u_em[2]

    for i in 1:N-1
        u[i+2] = u[i+1] + ((3/2)*h*f(t[i+1], u[i+1])) - ((1/2)*h*f(t[i], u[i]))
    end

    return u, t
end

function adams_bashforth_3rd_order(h::Float64, f::Function, y_0::Float64, M::Float64)
    N = ceil(Int64, M / h)
    u = zeros(Float64, N+1)
    t = range(0, M, step=h)

    # Set the initial condition.
    u[1] = y_0

	# Calculate y_1 using Euler's method first.
	u_em, _ = eulers_method(h, f, y_0, 2h)
	u[2] = u_em[2]
    u[3] = u_em[3]

    for i in 1:N-2
        u[i+3] = u[i+2] + h*((23/12)*f(t[i+2], u[i+2]) - (16/12)*f(t[i+1], u[i+1]) + (5/12)*f(t[i], u[i]))
    end

    return u, t
end

f(t, y) = y
y_true(t) = exp(t)

y_0 = 1.0
h = 0.25
T = 5.0

t = range(0, T, step=h)
u_eulers, t_eulers = eulers_method(h, f, y_0, T)
u_2step, _ = two_step_AB(h, f, y_0, T)
u_3step, _ = adams_bashforth_3rd_order(h, f, y_0, T)
u_true = y_true.(t)

plot(t, u_eulers, label="Euler's Method")
plot!(t, u_2step, label="2nd Order Adams-Bashforth")
plot!(t, u_3step, label="3rd Order")
plot!(t, u_true, label="exp(t)")
