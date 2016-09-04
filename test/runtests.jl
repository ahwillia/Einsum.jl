using Base.Test
using Einsum

## Test that vars in Main aren't overwritten by einsum
i = -1
y = randn(10)
@einsum x[i] := y[i] 
@test i == -1

## Test bounds checking on lhs
X = randn(10,11)
Y = randn(10,10)
Q = randn(9)

## Test bounds checking on rhs
B = randn(10,10)
A = randn(5,10)
@einsum B[i,j] := A[i,j] # this should run without a problem

## Test with preallocated array ##

A = zeros(5,6,7);
B = zeros(5,6,7);
X = randn(5,2);
Y = randn(6,2);
Z = randn(7,2);

@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
@einsimd B[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]

for i = 1:5
    for j = 1:6
        for k = 1:7
            s = 0.0
            for r = 1:2
                s += X[i,r]*Y[j,r]*Z[k,r]
            end
            @test isapprox(A[i,j,k],s)
            @test isapprox(B[i,j,k],s)
        end
    end
end

## Test without preallocated array ##

@einsum A2[i,j,k] := X[i,r]*Y[j,r]*Z[k,r]
@test isapprox(A,A2)

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
@test isapprox(k,dot(x,y))

# Elementwise multiplication (this should create nested loops with no
# no summation, due to lack of repeated variables.)
x = randn(10)
y = randn(10)
@einsum k[i] := x[i]*y[i]
@test isapprox(k,x.*y)

# Transpose a block matrix
z = Any[ rand(2,2) for i=1:2, j=1:2]
@einsum t[i,j] := transpose(z[j,i])
@test isapprox(z[1,1], t[1,1]')
@test isapprox(z[2,2], t[2,2]')
@test isapprox(z[1,2], t[2,1]')
@test isapprox(z[2,1], t[1,2]')

# Mapping functions
A = randn(10,10)
@einsum B[i,j] := exp(A[i,j])

# Example from numpy
A = reshape(collect(1:25),5,5)
@einsum B[i] := A[i,i]
@test all(B .== [1,7,13,19,25])

#
# TODO: consider adding support for this:
# @einsum A[i,j] = A[i,j] + 50
#

## Test in-place operation with preallocated array ##
A = randn(5,6,7);
B = randn(5,6,7);
A1 = copy(A);
B1 = copy(B);

X = randn(5,2);
Y = randn(6,2);
Z = randn(7,2);

@einsum A[i,j,k] += X[i,r]*Y[j,r]*Z[k,r]
@einsimd B[i,j,k] += X[i,r]*Y[j,r]*Z[k,r]

for i = 1:5
    for j = 1:6
        for k = 1:7
            s = 0.0
            for r = 1:2
                s += X[i,r]*Y[j,r]*Z[k,r]
            end
            @test isapprox(A[i,j,k],A1[i,j,k]+s)
            @test isapprox(B[i,j,k],B1[i,j,k]+s)
        end
    end
end

x = randn(10)
y = randn(10)
k0 = randn()
k = k0
@einsum k += x[i]*y[i]
@test isapprox(k,k0+dot(x,y))

# test multiplication

A1[:] = A[:]
B1[:] = B[:]

@einsum A[i,j,k] *= X[i,r]*Y[j,r]*Z[k,r]
@einsimd B[i,j,k] *= X[i,r]*Y[j,r]*Z[k,r]

for i = 1:5
    for j = 1:6
        for k = 1:7
            s = 0.0
            for r = 1:2
                s += X[i,r]*Y[j,r]*Z[k,r]
            end
            @test isapprox(A[i,j,k],A1[i,j,k]*s)
            @test isapprox(B[i,j,k],B1[i,j,k]*s)
        end
    end
end

x = randn(10)
y = randn(10)
k0 = randn()
k = k0
@einsum k *= x[i]*y[i]
@test isapprox(k,k0*dot(x,y))

# Test offsets

X = randn(10)
B = zeros(10)

@einsum A[i] := X[i+5]
@test size(A) == (5,)
@test all(A .== X[6:end])

@einsum B[i] = X[i+5]

@test size(B) == (10,)
@test all(B[1:5] .== X[6:end])
