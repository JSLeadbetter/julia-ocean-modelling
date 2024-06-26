# Helper functions.
eye(n::Int) = Matrix{Float64}(I, n, n)
speye(n::Int) = spdiagm(ones(n))
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

function zeros_like(A)
    m, n = size(A)
    return zeros(m, n)
end
