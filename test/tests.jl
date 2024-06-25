using Test

include("../adams_bashforth.jl")

@testset "Eulers Method" begin
    # Define parameters for our function.
	h = 0.5
    f(t, y) = y
	T = 5.0
    y_0 = 1.0

    # The true solution
    y_true(t) = exp(t)

    u, t = eulers_method(h, f, y_0, T)

    @test size(u)[1] == (11)
end

@testset "Adams-Bashforth 2nd Order" begin
    # Define parameters for our function.
	h = 0.5
    f(t, y) = y
	T = 5.0
    y_0 = 1.0

    # The true solution
    y_true(t) = exp(t)

    u, t = adams_bashforth_2nd_order(h, f, y_0, T)

    @test size(u)[1] == (11)
end
