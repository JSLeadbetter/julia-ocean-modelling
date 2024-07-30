# Implementing the Arakawa 1966 Jacobian finite differencing scheme.

using LinearAlgebra

include("../toolbox.jl")
include("BC.jl")

function j_pp(d::Float64, zeta::Matrix{Float64}, psi::Matrix{Float64})
    M, P = size(zeta)
    j_pp = zeros(M, P)

    for j in 2:P-1
        for i in 2:M-1
            j_pp[i, j] = (
                (zeta[i+1, j] - zeta[i-1, j])*(psi[i, j+1] - psi[i, j-1])
                - (zeta[i, j+1] - zeta[i, j-1])*(psi[i+1, j] - psi[i-1, j]))
        end
    end
    
    return (1 / (4d^2)) * j_pp
end

function j_pt(d::Float64, zeta::Matrix{Float64}, psi::Matrix{Float64})
    M, P = size(zeta)
    j_pt = zeros(M, P)

    for j in 2:P-1
        for i in 2:M-1
            j_pt[i, j] = (
                zeta[i+1, j]*(psi[i+1, j+1] - psi[i+1, j-1])
                - zeta[i-1, j]*(psi[i-1,j+1] - psi[i-1, j-1])
                - zeta[i, j+1]*(psi[i+1, j+1] - psi[i-1, j+1])
                + zeta[i, j-1]*(psi[i+1, j-1] - psi[i-1, j-1])
            )
        end
    end
    
    return (1 / (4d^2)) * j_pt
end

function j_tp(d::Float64, zeta::Matrix{Float64}, psi::Matrix{Float64})
    M, P = size(zeta)
    j_tp = zeros(M, P)

    for j in 2:P-1
        for i in 2:M-1
            j_tp[i, j] = (
                zeta[i+1, j+1]*(psi[i, j+1] - psi[i+1, j])
                - zeta[i-1, j-1]*(psi[i-1, j] - psi[i, j-1])
                - zeta[i-1, j+1]*(psi[i, j+1] - psi[i-1, j])
                + zeta[i+1, j-1]*(psi[i+1, j] - psi[i, j-1])
            )
        end
    end
    
    return (1 / (4d^2)) * j_tp
end

function J(d::Float64, zeta::Matrix{Float64}, psi::Matrix{Float64})
    j = (1/3) * (j_pp(d, zeta, psi) + j_pt(d, zeta, psi) + j_tp(d, zeta, psi))
    update_doubly_periodic_bc!(j)
    return j
end
