using Einsum

function benchmark_einsum(dim::Integer)
	X = zeros(dim,dim,dim,dim,dim)
	A = randn(dim,dim)
	B = randn(dim,dim)
	C = randn(dim,dim)
	D = randn(dim,dim)
	E = randn(dim,dim)

	return @einsum X[a,b,c,d,e] = A[r,a]*B[r,b]*C[r,c]*D[r,d]*E[r,e]
end

@time benchmark_einsum(30)
