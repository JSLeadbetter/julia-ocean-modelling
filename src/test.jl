using Test
using LinearAlgebra

include("schemes/helmholtz.jl")
include("model.jl")

@testset "Helmholtz solver" begin
    @testset "My Example" begin
        
        test_u(x, y) = sin((pi/10) * x) * sin((pi/10) * y)
        test_f(x, y) = -(pi^2 / 50) * sin((pi/10) * x) * sin((pi/10) * y)

        domain = RectangularDomain(0, 10, 0, 10)
        alpha = 0.0
        M_list = [8, 16, 32, 64]
        P_list = [8, 16, 32, 64]

        for i in eachindex(M_list)
            M = M_list[i]
            P = P_list[i]

            dx = (domain.x2 - domain.x1) / M
            xs = range(domain.x1+dx, domain.x2-dx, length=M+2)
            ys = range(domain.y1+dx, domain.y2-dx, length=P+2)

            u = sp_solve_modified_helmholtz(M, P, dx, test_f, alpha, domain)
            u_true = inflate(test_u, xs, ys)
            error = dx * norm(u - u_true)

            println("M = $M, Norm = $error")
        end
    end
    # @testset "NPDEs Example"
end

@testset "P and P inverse correct" begin
    KM = 1000.0

    H_1 = 1000.0KM
    H_2 = 1000.0KM
    
    P = P_matrix(H_1, H_2)
    P_inv = P_inv_matrix(H_1, H_2)

    x = P * P_inv
    @test x == diagm(ones(2))
end

@testset "Poisson problem Laplacian A matrix construction" begin
    M = 4
    P = 3
    alpha = 0.0
    dx = 1.0
    A = construct_spA(M, P, dx, alpha)

    @test isposdef(-A)
end

@testset "Construction of 1D Laplacian matrix" begin
    N = 4
    lap = laplacian_1d(N)
    
    expected_lap = [[-2.0, 1, 0, 1] [1, -2, 1, 0] [0, 1, -2, 1] [1, 0, 1, -2]]
    
    @test size(expected_lap) == size(lap)
    
    @test expected_lap == Matrix(lap)
end

@testset "Construction of 2D Laplacian matrix" begin
    M = 3
    P = 3
    lap = laplacian_2d(M, P)
end

@testset "Cholesky factorisation of A" begin
    @testset "4x4 Square" begin
        M = P = 4
        alpha = 1.0
        dx = 1.0
        A = -construct_spA(M, P, dx, alpha)

        A[:,1] .= 0
        A[1,:] .= 0
        A[1, 1] = 1

        @test typeof(A) == SparseMatrixCSC{Float64, Int64}
        @test isposdef(A)
        @test issymmetric(A)
    end
    @testset "10x5 Rectangle" begin
        M = P = 5
        alpha = 1.0
        dx = 1.0
        A = -construct_spA(M, P, dx, alpha)

        A[:,1] .= 0
        A[1,:] .= 0
        A[1, 1] = 1

        @test isposdef(A)
        @test issymmetric(A)
    end
end

@testset "Chol. Fact. Poisson" begin
    @testset "4 x 4 square" begin
        M = 4
        P = 4
        dx = 1.0
        A = get_poisson_chol_A(M, P, dx)
    end
    @testset "8 x 4 rectangle" begin
        M = 8
        P = 4
        dx = 1.0
        A = get_poisson_chol_A(M, P, dx)
    end
end

@testset "Chol. Fact. Helmholtz" begin
    @testset "4 x 4 square" begin
        M = 4
        P = 4
        dx = 1.0
        alpha = 3.0
        A = get_helmholtz_chol_A(M, P, dx, alpha)
    end
    @testset "8 x 4 rectangle" begin
        M = 8
        P = 4
        dx = 1.0
        alpha = 3.0
        A = get_helmholtz_chol_A(M, P, dx, alpha)
    end
end

