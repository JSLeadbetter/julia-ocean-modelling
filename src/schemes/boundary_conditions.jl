"""Update the ghost cells of a matrix for periodic boundary conditions."""
function update_doubly_periodic_bc!(b::Matrix{Float64})
    b[2:end-1,1] = b[2:end-1,end-1] # LHS ghost column.
    b[2:end-1,end] = b[2:end-1,2] # RHS ghost column.
    b[1,2:end-1] = b[end-1,2:end-1] # Top ghost row.
    b[end,2:end-1] = b[2,2:end-1] # Bottom ghost row.

    # Corner cells need to be copied diagonally.
    b[1, 1] = b[end-1, end-1]
    b[1, end] = b[end-1, 2]
    b[end, end] = b[2, 2]
    b[end, 1] = b[2, end-1]
end

"""Extend a matrix by two rows and columns and copies rows/cols to add double-periodicity."""
function add_doubly_periodic_boundaries(u::Matrix{Float64})
    M, P = size(u)
    extended_u = zeros(M+2, P+2)
    extended_u[2:end-1, 2:end-1] = u
    update_doubly_periodic_bc!(extended_u)
    return extended_u
end
