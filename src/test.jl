using Test
using LinearAlgebra
using CurveFit
using Plots

include("schemes/helmholtz.jl")
include("schemes/arakawa.jl")
include("model.jl")

const MINUTES = 60
const DAY = 60*60*24
const KM = 1000.0
const YEAR = 60*60*24*365

@testset "Parameter Values" begin
    H_1 = 1.0*KM
    H_2 = 2.0*KM
    beta = 2*10^-11
    Lx = 4000.0*KM
    Ly = 4000.0*KM
    dt = 15.0*MINUTES
    T = 0.5YEAR
    U = 2.0
    M = P = 128
    dx = Lx / M
    visc = 100.0
    r = 10^-7
    R_d = 40.0*KM
    initial_kick = 1e-2
    model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d, initial_kick)

    expected_ratio = 0.5 * (1000 + 2000) / (40000^2 * (1/1000 + 1/2000)) 
    @test expected_ratio == ratio_term(model)
    
    expected_S1_plus = 2expected_ratio / (1000 * 3000)
    @test expected_S1_plus == S1_plus(model)

    expected_S2_minus = 2expected_ratio / (2000 * 3000)
    @test expected_S2_minus == S2_minus(model)

    expected_beta_1 = beta + expected_S1_plus * U
    @test expected_beta_1 == beta_1(model)

    expected_beta_2 = beta - expected_S2_minus * U
    @test expected_beta_2 == beta_2(model)

    expected_s_eig = -1 / R_d^2
    @test S_eig(model) == expected_s_eig

    @test (-S1_plus(model) - S2_minus(model)) == expected_s_eig
end

@testset "Centred Difference Convergence" begin
    f(x, y) = x^3 + y^2
    xs = 1:10
    ys = 1:10
    u = Matrix{Float64}(inflate(f, xs, ys))
    dx = 1.0

    # true_ux
end


@testset "Explicit Laplacian Convergence" begin
    f(x, y) = x^3 + y^2
    xs = 1:10
    ys = 1:10
    u = Matrix{Float64}(inflate(f, xs, ys))
    dx = 1.0

    true_lap_f(x, y) = 6x + 2
    true_lap = Matrix{Float64}(inflate(true_lap_f, xs, ys))
    update_doubly_periodic_bc!(true_lap)

    lap = laplace_5p(u, dx)

    @test lap == true_lap
end

@testset "Arakawa Convergence" begin
    Lx = 10
    Ly = 10
    A_func(x, y) = sin(2pi*x/Lx)*sin(2pi*y/Ly)
    B_func(x, y) = cos(2pi*x/Lx)*cos(2pi*y/Ly) 
    true_J_func(x, y) = -4pi^2/(Lx*Ly)*cos(2pi*x/Lx)^2*sin(2pi*y/Ly)^2 + 4pi^2/(Lx*Ly)*sin(2pi*x/Lx)^2*cos(2pi*y/Ly)^2 
    
    # M = P = 16
    M_list = [8, 16, 32, 64, 128, 256]
    errors = zeros(size(M_list)[1])

    for (i, M) in enumerate(M_list)
        dx = Lx / M
        P = M

        xs = range(-dx, Lx, length=M+2)
        ys = range(-dx, Ly, length=P+2)
        A = Matrix{Float64}(inflate(A_func, xs, ys))
        B = Matrix{Float64}(inflate(B_func, xs, ys))

        true_J_matrix = inflate(true_J_func, xs, ys)
        test_J_matrix = J(dx, A, B)

        errors[i] = dx * norm(test_J_matrix - true_J_matrix)
    end

    fit = linear_fit(log.(M_list), log.(errors))
    slope = fit[2]

    println(M_list)
    println(errors)
    println("Log-log Slope = ", slope)
end

@testset "Doubly Periodic Poisson Solve" begin
    inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]
    
    x_0 = 0
    x_1 = 3
    y_0 = 0
    y_1 = 3
    Lx = x_1 - x_0
    Ly = y_1 - y_0

    u(x, y) = sin(2pi * (x) / Lx) * cos(2pi * (y) / Ly)
    
    # Uxx + Uyy = f(x, y)
    f(x, y) = -(pi^2)*(u(x,y)*(4/Ly^2 + 4/Lx^2))

    alpha = 0.0
    M_list = [4, 8, 16, 32, 64]
    errors = zeros(size(M_list)[1])
    
    for (i, M) in enumerate(M_list)
        P = M
        dx = Lx / M

        xs = range(x_0 - dx, x_1, length=M+2)
        ys = range(y_0 - dx, y_1, length=P+2)
        b = inflate(f, xs, ys)
        
        u_num = sp_solve_modified_helmholtz(M, P, dx, b, alpha)
        u_true = inflate(u, xs, ys)

        # Weighted 2-norm.
        errors[i] = dx * norm(u_num - u_true)
    end

    # Calculate the slope of log-log values to check for second order convergence.
    fit = linear_fit(log.(M_list), log.(errors))
    slope = fit[2]

    println(M_list)
    println(errors)
    println("Log-log Slope = ", slope)

    @test -slope > 1.7 && -slope < 2.3
end

@testset "Doubly Periodic Helmholtz Solve" begin
    inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]
    
    x_0 = 0
    x_1 = 3
    y_0 = 0
    y_1 = 3
    Lx = x_1 - x_0
    Ly = y_1 - y_0

    alpha = -3.0
    u(x, y) = sin(2pi * (x) / Lx) * cos(2pi * (y) / Ly)
    
    # Uxx + Uyy + 3u = f(x, y)
    f(x, y) = -(pi^2)*(u(x,y)*(4/Ly^2 + 4/Lx^2)) + alpha*u(x, y)
    
    M_list = [4, 8, 16, 32, 64]
    errors = zeros(size(M_list)[1])
    
    for (i, M) in enumerate(M_list)
        P = M
        dx = Lx / M

        xs = range(x_0 - dx, x_1, length=M+2)
        ys = range(y_0 - dx, y_1, length=P+2)
        b = inflate(f, xs, ys)
        
        u_num = sp_solve_modified_helmholtz(M, P, dx, b, alpha)
        u_true = inflate(u, xs, ys)

        # Weighted 2-norm.
        errors[i] = dx * norm(u_num - u_true)
    end

    # Calculate the slope of log-log values to check for second order convergence.
    fit = linear_fit(log.(M_list), log.(errors))
    slope = fit[2]

    println(M_list)
    println(errors)
    println("Log-log Slope = ", slope)

    @test -slope > 1.7 && -slope < 2.3
end

@testset "P and P Inverse Correct" begin
    H_1 = 1.0*KM
    H_2 = 2.0*KM
    beta = 2*10^-11
    Lx = 4000.0*KM
    Ly = 4000.0*KM
    dt = 15.0*MINUTES
    T = 0.5YEAR
    U = 2.0
    M = P = 128
    dx = Lx / M
    visc = 100.0
    r = 10^-7
    R_d = 40.0*KM
    initial_kick = 1e-2
    model = BaroclinicModel(H_1, H_2, beta, Lx, Ly, dt, T, U, M, P, dx, visc, r, R_d, initial_kick)
    
    P = P_matrix(H_1, H_2)
    P_inv = P_inv_matrix(model)

    x = P * P_inv
    @test x == diagm(ones(2))
end

@testset "Poisson Problem Laplacian A Matrix Construction" begin
    M = 4
    P = 3
    alpha = 0.0
    dx = 1.0
    A = -construct_spA(M, P, dx, alpha)

    @test isposdef(A)
end

@testset "Construction of 1D Periodic Laplacian Matrix" begin
    N = 4
    lap = laplacian_1d_periodic(N)
    
    expected_lap = [[-2.0, 1, 0, 1] [1, -2, 1, 0] [0, 1, -2, 1] [1, 0, 1, -2]]
    
    @test size(expected_lap) == size(lap)
    
    @test expected_lap == Matrix(lap)
end

@testset "Construction of 2D Laplacian Matrix" begin
    M = 3
    P = 3
    lap = laplacian_2d(M, P)
end

@testset "Modified Helmholtz matrix is pos-def" begin
    @testset "4x4 Square" begin
        M = P = 4
        alpha = -3.0
        dx = 0.5
        A = -construct_spA(M, P, dx, alpha)

        A[:,1] .= 0
        A[1,:] .= 0
        A[1, 1] = 1

        @test typeof(A) == SparseMatrixCSC{Float64, Int64}
        @test issymmetric(A)
        @test isposdef(A)
    end
    @testset "10x5 Rectangle" begin
        M = 10
        P = 5
        alpha = -1.0
        dx = 1.0
        A = -construct_spA(M, P, dx, alpha)

        A[:,1] .= 0
        A[1,:] .= 0
        A[1, 1] = 1

        @test typeof(A) == SparseMatrixCSC{Float64, Int64}
        @test issymmetric(A)
        @test isposdef(A)
    end
end
