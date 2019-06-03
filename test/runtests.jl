using Test
using LinearAlgebra # dot

using Einsum

@testset "Test that vars in Main aren't overwritten by einsum" begin
    i = -1
    y = randn(10)
    @einsum x[i] := y[i]
    @test i == -1
end

@testset "Test that B is overwritten by := operator" begin
    B = randn(10, 10)
    A = randn(5, 10)
    @einsum B[i, j] := A[i, j] # this should run without a problem
    @test size(B) == size(A)
end

@testset "CP decomposition test case" begin

    # preallocated test case
    A = zeros(5, 6, 7)
    B = similar(A)
    C = similar(A)

    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)

    @einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    @einsimd B[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    @vielsum C[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]

    for i = 1:5
        for j = 1:6
            for k = 1:7
                s = 0.0
                for r = 1:2
                    s += X[i, r] * Y[j, r] * Z[k, r]
                end
                @test isapprox(A[i, j, k], s)
                @test isapprox(B[i, j, k], s)
                @test isapprox(C[i, j, k], s)
            end
        end
    end

    # without preallocation
    @einsum A2[i, j, k] := X[i, r] * Y[j, r] * Z[k, r]
    @test isapprox(A, A2)

end

@testset "Interesting test case, can throw an error that local vars are declared twice." begin
    A = zeros(5, 6, 7)
    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)
    if true
        @einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    else
        @einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
    end
end

@testset "From #21: local `T` does not interfer with internal T" begin
    function test(x::Vector{T}, y::Vector{T}) where T
        @einsum z := x[i] * y[i]
        return z
    end
    @test_nowarn test(rand(3), rand(3))
end

@testset "From #20: local `s` does not interfere with internal s" begin
    x = rand(2, 3)
    @test_nowarn @einsum y[i] := x[i, s]
end

@testset "At one point this threw an error because the lhs had no indices/arguments" begin
    x = randn(10)
    y = randn(10)
    @einsum k := x[i] * y[i]
    @test isapprox(k, dot(x, y))
end

@testset "Elementwise multiplication (this should create nested loops with no no summation.)" begin
    x = randn(10)
    y = randn(10)
    @einsum k[i] := x[i] * y[i]
    @einsimd k2[i] := x[i] * y[i]
    @vielsum k3[i] := x[i] * y[i]
    @test isapprox(k, x .* y)
    @test isapprox(k2, x .* y)
    @test isapprox(k3, x .* y)
end

@testset "Transpose a block matrix" begin
    z = [rand(2, 2) for i = 1:2, j = 1:2]
    @einsum t[i, j] := transpose(z[j, i])
    @test isapprox(z[1, 1],  t[1, 1]')
    @test isapprox(z[2, 2],  t[2, 2]')
    @test isapprox(z[1, 2],  t[2, 1]')
    @test isapprox(z[2, 1],  t[1, 2]')
end

@testset "Mapping functions" begin
    A = randn(10, 10)
    @einsum B[i, j] := exp(A[i, j])
    @test isapprox(exp.(A), B)
end

@testset "Example from numpy" begin
    A = reshape(collect(1:25), 5, 5)
    @einsum B[i] := A[i, i]
    @test all(B .== [1, 7, 13, 19, 25])

    @vielsum C[i] := A[i, i]
    @test all(C .== [1, 7, 13, 19, 25])

end

@testset "Adding a scalar, to-done" begin

    A = collect(reshape(1:12, 3,4))
    @einsum A[i, j] = A[i,j] + 50
    @test A == collect(50 .+ reshape(1:12, 3,4))

end

@testset "Test in-place operations" begin
    A = randn(5, 6, 7)
    B = randn(5, 6, 7)
    A1 = copy(A)
    B1 = copy(B)

    X = randn(5, 2)
    Y = randn(6, 2)
    Z = randn(7, 2)

    @einsum A[i, j, k] += X[i, r] * Y[j, r] * Z[k, r]
    @einsimd B[i, j, k] += X[i, r] * Y[j, r] * Z[k, r]

    for i = 1:5
        for j = 1:6
            for k = 1:7
                s = 0.0
                for r = 1:2
                    s += X[i, r] * Y[j, r] * Z[k, r]
                end
                @test isapprox(A[i, j, k], A1[i, j, k] + s)
                @test isapprox(B[i, j, k], B1[i, j, k] + s)
            end
        end
    end

    x = randn(10)
    y = randn(10)
    k0 = randn()
    k = k0
    @einsum k += x[i] * y[i]
    @test isapprox(k, k0 + dot(x, y))

    # test multiplication

    A1[:] = A[:]
    B1[:] = B[:]

    @einsum A[i, j, k]  *= X[i, r] * Y[j, r] * Z[k, r]
    @einsimd B[i, j, k] *= X[i, r] * Y[j, r] * Z[k, r]

    for i = 1:5
        for j = 1:6
            for k = 1:7
                s = 0.0
                for r = 1:2
                    s += X[i, r] * Y[j, r] * Z[k, r]
                end
                @test isapprox(A[i, j, k], A1[i, j, k] * s)
                @test isapprox(B[i, j, k], B1[i, j, k] * s)
            end
        end
    end

    x = randn(10)
    y = randn(10)
    k0 = randn()
    k = k0
    @einsum k *= x[i] * y[i]
    @test isapprox(k, k0 * dot(x, y))
end

@testset "Test offsets" begin
    X = randn(10)

    # without preallocation
    @einsum A[i] := X[i + 5]
    @test size(A) == (5,)
    @test all(A .== X[6:end])

    # with preallocation
    B = zeros(10)
    @einsum B[i] = X[i + 5]
    @test size(B) == (10,)
    @test all(B[1:5] .== X[6:end])
end

@testset "Test symbolic offsets" begin
    offset = 5
    X = randn(10)

    # without preallocation
    # @einsum A[i] := X[i + :offset] # error on 1.0
    # @test size(A) == (5,)
    # @test all(A .== X[6:end])

    # with preallocation
    B = zeros(10)
    # @einsum B[i] = X[i + :offset] # error on 1.0
    # @test size(B) == (10,)
    # @test all(B[1:5] .== X[6:end])
end

@testset "Test adding/subtracting constants" begin
    k = 5
    X = randn(10)

    # without preallocation
    @einsum A[i] := X[i] + k
    @einsum B[i] := X[i] - k
    @test isapprox(A, X .+ k)
    @test isapprox(B, X .- k)

    # with preallocation
    C, D = zeros(10), zeros(10)
    @einsum C[i] = X[i] + k
    @einsum D[i] = X[i] - k
    @test isapprox(C, X .+ k)
    @test isapprox(D, X .- k)
end

@testset "Test multiplying/dividing constants" begin
    k = 5
    X = randn(10)

    # without preallocation
    @einsum A[i] := X[i] * k
    @einsum B[i] := X[i] / k
    @test isapprox(A, X .* k)
    @test isapprox(B, X ./ k)

    # with preallocation
    C, D = zeros(10), zeros(10)
    @einsum C[i] = X[i] * k
    @einsum D[i] = X[i] / k
    @test isapprox(C, X .* k)
    @test isapprox(D, X ./ k)
end

@testset "Test indexing with a constant" begin
    A = randn(10, 2)
    j = 2
    # @einsum B[i] := A[i, :j] # error on 1.0
    # @test all(B .== A[:, j])
    @einsum C[i] := A[i, 1]
    @test all(C .== A[:, 1])

    D = zeros(10, 3)
    # @einsum D[i, 1] = A[i, :j]
    # @test isapprox(D[:, 1], A[:, j])
    # @einsum D[i, :j] = A[i, :j]
    # @test isapprox(D[:, j], A[:, j])
end

@testset "Better type inference on allocating arrays" begin
    B1 = ones(Int, 5)
    B2 = ones(Float32, 5)
    B3 = ones(5)
    C = randn(5)
    @einsum A1[i, j] := B1[i] * C[j]
    @einsum A2[i, j] := B2[i] * C[j]
    @einsum A3[i, j] := B3[i] * C[j]

    @test eltype(A1) == Float64
    @test eltype(A2) == Float64
    @test eltype(A3) == Float64
    @test isapprox(A1, A3)
    @test isapprox(A2, A3)
end
