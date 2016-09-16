# Einsum.jl
Einstein summation notation similar to numpy's [`einsum`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html) function (but more flexible!).

| **PackageEvaluator** | **Package Build** | **Package Status** |
|:--------------------:|:---------:|:------------------:|
| [![Einsum](http://pkg.julialang.org/badges/Einsum_0.5.svg)](http://pkg.julialang.org/?pkg=Einsum) | [![Build Status](https://travis-ci.org/ahwillia/Einsum.jl.svg?branch=master)](https://travis-ci.org/ahwillia/Einsum.jl) | [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | 
[![Einsum](http://pkg.julialang.org/badges/Einsum_0.4.svg)](http://pkg.julialang.org/?pkg=Einsum) | | [![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) |

To install: `Pkg.add("Einsum")`.

## Documentation

### Quickstart example:

To install and load the package use:
```julia
Pkg.add("Einsum")
using Einsum
```
This package exports a single macro `@einsum`, which implements *similar* notation to the [Einstein summation convention](https://en.wikipedia.org/wiki/Einstein_notation) to flexibly specify operations on Julia `Array`s. For example, basic matrix multiplication can be implemented as:
```julia
@einsum A[i,j] := B[i,k]*C[k,j]
```
To execute this computation, `@einsum` uses [Julia's metaprogramming capabilities](http://docs.julialang.org/en/stable/manual/metaprogramming/) to generate and execute a series of nested for loops. In a nutshell, the generated code looks like this:
```julia
# determine type
T = promote_type(eltype(B),eltype(C))

# allocate new array
A = Array(T,size(B))

# check dimensions
@assert size(B,2) == size(C,2)

# main loop
@inbounds begin # skip bounds-checking for speed
    for i = 1:size(B,1), j = 1:size(C,2)
        s = zero(T)
        for k = 1:size(B,2)
            s += B[i,k]*C[k,j]
        end
        A[i,j] = z
    end
end
```
The actual generated code is a bit more verbose (and not neatly commented/formatted); you can view it by using the [`macroexpand`](http://docs.julialang.org/en/stable/stdlib/base/#Base.macroexpand) command:
```julia
macroexpand(:( @einsum A[i,j] := B[i,k]*C[k,j] ))
```

### Assignment and updating operators:

* `=` overwrite result in existing Array
* `:=` allocate a new array
* `+=`, `-=` add to or subtract from an existing array 

Calling `@einsum` with the `:=` operator separating the left and right hand side of the equation signals that you'd like to allocate a new array to store the result of the computation. This borrows from mathematical notation where `:=` denotes "equal to by definition." An example:
```julia
using Einsum
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)

# creates new array A with appropriate dimensions
@einsum A[i,j,k] := X[i,r]*Y[j,r]*Z[k,r]
```
If you would rather allocate the space yourself, use the `=` operator instead
```julia
using Einsum
A = randn(5,6,7); # preallocate space yourself
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)

# Store the result in A, overwriting as necessary
@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
```

### Rules for indexing variables

* Indices that show up on the left-hand-side but not the right-hand-side are summed over
* Indices that appear over multiple dimensions must match

`@einsum` iterates over the extent of the right-hand-side indices. For example, the following code allocates an array `A` that is the same size as `B` and copies its data into `A`:
```julia
@einsum A[i,j] := B[i,j]  # same as A = copy(B)
```
If an index appears on the right-hand-side, but does not appear on the left-hand-side, then this variable is summed over. For example, the following code allocates `A` to be `size(B,1)` and sums over the rows of `B`:
```julia
@einsum A[i] := B[i,j]  # same as A = sum(B,2)
```
If an index variable appears multiple times on the right-hand-side, then it is asserted that the sizes of these dimensions match. For example,
```julia
@einsum A[i] := B[i,j]*C[j]
```
will check that the second dimension of `B` matches the first dimension of `C` in length. In particular it is equivalent to the following code:
```
A = zeros(size(B,1))
@assert size(B,2) == size(C,1)
for i = 1:size(B,1), j = 1:size(B,2)
    A[i] += B[i,j]*C[j]
end
```
So an error will be thrown if the specified dimensions of `B` and `C` don't match.

### Advanced indexing -- symbols

The following example will copy the fifth column of `B` into `A`.
```julia
j = 5
@einsum A[i] = B[i,:j]
```

### Advanced indexing -- shifts and offsets

`@einsum` also allows offsets on the right-hand-side:
```julia
@einsum A[i] = B[i-5]
```
Symbolic offsets are also possible:
```julia
j = 5
@einsum A[i] = B[i-:j]
```

### Related Packages:

* https://github.com/Jutho/TensorOperations.jl
