# Einsum.jl
Einstein summation notation similar to numpy's [`einsum`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html) function (but more flexible!).

| **PackageEvaluator** | **Package Build** | **Package Status** |
|:--------------------:|:---------:|:------------------:|
| [![Einsum](http://pkg.julialang.org/badges/Einsum_0.5.svg)](http://pkg.julialang.org/?pkg=Einsum) | [![Build Status](https://travis-ci.org/ahwillia/Einsum.jl.svg?branch=master)](https://travis-ci.org/ahwillia/Einsum.jl) | [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | 
[![Einsum](http://pkg.julialang.org/badges/Einsum_0.4.svg)](http://pkg.julialang.org/?pkg=Einsum) | | [![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active) |

To install: `Pkg.add("Einsum")`.

## New Documentation:

## Documentation:

#### If destination is preallocated use `=`

```julia
using Einsum
A = zeros(5,6,7); # need to preallocate destination
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
```

#### If destination is not preallocated use `:=`

```julia
using Einsum
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum A[i,j,k] := X[i,r]*Y[j,r]*Z[k,r] # creates new array A with appropriate dimensions
```

#### What happens under the hood:

The `@einsum` macro automatically generates code that looks much like the following (note that we "sum out" over the index `r`, since it only occurs on the right hand side of the equation):

```julia
for k = 1:size(A,3)
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            s = 0
            for r = 1:size(X,2)
                s += X[i,r] * Y[j,r] * Z[k,r]
            end
            A[i,j,k] = s
        end
    end
end
```

To see exactly what is generated, use [`macroexpand`](http://docs.julialang.org/en/release-0.4/manual/metaprogramming/#macros):

```julia
macroexpand(:(@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]))
```

### Other functionality

In principle, the `@einsum` macro can flexibly implement function calls within the nested for loop structure. For example, consider transposing a block matrix:

```julia
z = Any[ rand(2,2) for i=1:2, j=1:2]
@einsum t[i,j] := transpose(z[j,i])
```

This produces a for loop structure with a `transpose` function call in the middle. Approximately:

```
for j = 1:size(z,1)
    for i = 1:size(z,2)
        t[i,j] = transpose(z[j,i])
    end
end
```

Again, you can use [`macroexpand`](http://docs.julialang.org/en/release-0.4/manual/metaprogramming/#macros) to see the exact code that is generated.

### Related Packages:

* https://github.com/Jutho/TensorOperations.jl
