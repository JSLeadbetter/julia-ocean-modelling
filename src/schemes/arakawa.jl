"""Implementation of the Arakawa 1966 Jacobian finite differencing scheme."""

using LinearAlgebra

include("boundary_conditions.jl")

function j_pp(zeta::Matrix{Float64}, psi::Matrix{Float64})
    M, P = size(zeta)
    j_pp = zeros(M, P)

    @inbounds for j in 2:P-1
        @inbounds for i in 2:M-1
            j_pp[i, j] = (
                (zeta[i+1, j] - zeta[i-1, j])*(psi[i, j+1] - psi[i, j-1])
                - (zeta[i, j+1] - zeta[i, j-1])*(psi[i+1, j] - psi[i-1, j]))
        end
    end
    
    return j_pp
end

function j_pt(zeta::Matrix{Float64}, psi::Matrix{Float64})
    M, P = size(zeta)
    j_pt = zeros(M, P)

    @inbounds for j in 2:P-1
        @inbounds for i in 2:M-1
            j_pt[i, j] = (
                zeta[i+1, j]*(psi[i+1, j+1] - psi[i+1, j-1])
                - zeta[i-1, j]*(psi[i-1,j+1] - psi[i-1, j-1])
                - zeta[i, j+1]*(psi[i+1, j+1] - psi[i-1, j+1])
                + zeta[i, j-1]*(psi[i+1, j-1] - psi[i-1, j-1])
            )
        end
    end
    
    return j_pt
end

function j_tp(zeta::Matrix{Float64}, psi::Matrix{Float64})
    M, P = size(zeta)
    j_tp = zeros(M, P)

    @inbounds for j in 2:P-1
        @inbounds for i in 2:M-1
            j_tp[i, j] = (
                zeta[i+1, j+1]*(psi[i, j+1] - psi[i+1, j])
                - zeta[i-1, j-1]*(psi[i-1, j] - psi[i, j-1])
                - zeta[i-1, j+1]*(psi[i, j+1] - psi[i-1, j])
                + zeta[i+1, j-1]*(psi[i+1, j] - psi[i, j-1])
            )
        end
    end
    
    return j_tp
end

function J(dx::Float64, zeta::Matrix{Float64}, psi::Matrix{Float64})
    j = (j_pp(zeta, psi) + j_pt(zeta, psi) + j_tp(zeta, psi)) / (3 * 4 * dx^2)
    update_doubly_periodic_bc!(j)
    return j
end
