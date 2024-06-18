# Helper functions.
eye(n::Int) = Matrix{Float64}(I, n, n)
speye(n::Int) = spdiagm(ones(n))
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]