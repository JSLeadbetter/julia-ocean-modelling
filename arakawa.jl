# Implementing the Arakawa 1966 Jacobian finite differencing scheme.

using LinearAlgebra

include("toolbox.jl")

function j_pp(d::Float64, zeta::Matrix{Float64}, phi::Matrix{Float64})
    M, P = size(u)
    j_pp = zeros(M, P)

    for i in 2:M-1
        for j in 2:P-1
            j_pp[i, j] = (1/4d^2)*(
                (zeta[i+1, j] - zeta[i-1, j])*(phi[i, j+1] - phi[i, j-1])
                - (zeta[i, j+1] - zeta[i, j-1])*(phi[i+1, j] - phi[i-1, j]))
        end
    end
    
    return j_pp
end

function j_pt(d::Float64, zeta::Matrix{Float64}, phi::Matrix{Float64})
    M, P = size(u)
    j_pt = zeros(M, P)

    for i in 2:M-1
        for j in 2:P-1
            j_pt[i, j] = (1/4d^2)*(
                zeta[i+1, j]*(phi[i+1, j+1] - phi[i+1, j-1])
                - zeta[i-1, j]*(phi[i-1,j+1] - phi[i-1, j-1])
                - zeta[i, j+1]*(phi[i+1, j+1] - phi[i-1, j+1])
                + zeta[i, j-1]*(phi[i+1, j-1] - phi[i-1, j-1])
            )
        end
    end
    
    return j_pt
end

function j_tp(d::Float64, zeta::Matrix{Float64}, phi::Matrix{Float64})
    M, P = size(u)
    j_tp = zeros(M, P)

    for i in 2:M-1
        for j in 2:P-1
            j_tp[i, j] = (1/4d^2)*(
                zeta[i+1, j+1]*(phi[i, j+1] - phi[i+1, j])
                - zeta[i-1, j-1]*(phi[i-1, j] - phi[i, j-1])
                - zeta[i-1, j+1]*(phi[i, j+1] - phi[i-1, j])
                + zeta[i+1, j-1]*(phi[i+1, j] - phi[i, j-1])
            )
        end
    end
    
    return j_tp
end

function J(d::Float64, zeta::Matrix{Float64}, phi::Matrix{Float64})
    return (1/3) * (j_pp(d, zeta, phi) + j_pt(d, zeta, phi) + j_tp(d, zeta, phi))
end
