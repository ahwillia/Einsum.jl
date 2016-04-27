using Base.Test
using Einsum

## Test with preallocated array ##

A = zeros(5,6,7);
X = randn(5,2);
Y = randn(6,2);
Z = randn(7,2);

@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]

for i = 1:5
    for j = 1:6
        for k = 1:7
            s = 0
            for r = 1:2
                s += X[i,r]*Y[j,r]*Z[k,r]
            end
            @test A[i,j,k] == s
        end
    end
end

## Test without preallocated array ##

@einsum A2[i,j,k] := X[i,r]*Y[j,r]*Z[k,r]
@test all(A .== A2)

# Interesting test case, can throw an error that
# local vars are declared twice. Solution was to wrap
# everything in a let statement.
if true
    @einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
else
    @einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
end

# At one point this threw an error because the lhs
# had no indices/arguments
x = randn(10)
y = randn(10)
@einsum k := x[i]*y[i]
@test k == dot(x,y)

# Elementwise multiplication (this should create nested loops with no
# no summation, due to lack of repeated variables.)
x = randn(10)
y = randn(10)
@einsum k[i] := x[i]*y[i]
@test all(k .== x.*y)

# Transpose a block matrix
z = Any[ rand(2,2) for i=1:2, j=1:2]
@einsum t[i,j] := transpose(z[j,i])
@test all(z[1,1] .== t[1,1]')
@test all(z[2,2] .== t[2,2]')
@test all(z[1,2] .== t[2,1]')
@test all(z[2,1] .== t[1,2]')
