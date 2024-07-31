"""Update the ghost cells of a matrix for periodic boundary conditions."""
function update_doubly_periodic_bc!(b::Matrix{Float64})
    b[:,1] = b[:,end-1] # LHS ghost column.
    b[:,end] = b[:,2] # RHS ghost column.
    b[1,:] = b[end-1,:] # Top ghost row.
    b[end,:] = b[2,:] # Bottom ghost row.
end
