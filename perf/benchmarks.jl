using Einsum
using BenchmarkTools

srand(1234)
benchmarks = BenchmarkGroup()

### CP DECOMPOSITION BENCHMARK ###
"""
Composes a 5th-order tensor from 5 factors. This is a
CP decomposition, PARAFAC, or CANDECOMP decomposition of
tensors.
"""
function cpd(dim,r)
    A = randn(dim,r)
    B = randn(dim,r)
    C = randn(dim,r)
    D = randn(dim,r)
    E = randn(dim,r)
    @einsum X[a,b,c,d,e] := A[r,a]*B[r,b]*C[r,c]*D[r,d]*E[r,e]
    return X
end
benchmarks["cpd"] = @benchmarkable cpd(1000,10)

### COMMON FATE BENCHMARK ###
"""
Common fate bechmark proposed by @faroit
"""
function commonfate(dim,r)
    A = randn(r,dim,dim,dim)
    H = randn(r,dim)
    C = randn(r,dim)
    @einsum P[a,b,f,t,c] := A[j,a,b,f]*H[j,t]*C[j,c]
    return P
end
benchmarks["commonfate"] = @benchmarkable commonfate(30,10)



