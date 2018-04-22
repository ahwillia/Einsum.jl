# Einsum.jl
Einstein summation notation similar to numpy's [`einsum`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html) function (but more flexible!).

| **PackageEvaluator** | **Package Build** | **Package Status** |
|:--------------------:|:---------:|:------------------:|
| [![Einsum](http://pkg.julialang.org/badges/Einsum_0.7.svg)](http://pkg.julialang.org/?pkg=Einsum) | [![Build Status](https://travis-ci.org/ahwillia/Einsum.jl.svg?branch=master)](https://travis-ci.org/ahwillia/Einsum.jl) | [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | 
[![Einsum](http://pkg.julialang.org/badges/Einsum_0.6.svg)](http://pkg.julialang.org/?pkg=Einsum) | | [![Project Status: Inactive - The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive) - help wanted! |

To install: `Pkg.add("Einsum")`.

## Documentation

### Basics

If the destination array is preallocated, then use `=`:

```julia
A = zeros(5,6,7) # need to preallocate destination
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
```

If destination is not preallocated, then use `:=` to automatically create a new array A with appropriate dimensions:

```julia
using Einsum
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum A[i,j,k] := X[i,r]*Y[j,r]*Z[k,r]
```

### What happens under the hood

To see exactly what is generated, use [`@macroexpand`](https://docs.julialang.org/en/stable/stdlib/base/#Base.@macroexpand):

```julia
@macroexpand @einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
```

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

In reality, this code will be preceded by the the neccessary bounds checks and allocations, and take care to use the right types and keep hygenic.

You can also use updating assignment operators for preallocated arrays.  For example, `@einsum A[i,j,k] *= X[i,r]*Y[j,r]*Z[k,r]` will produce something like

```julia
for k = 1:size(A,3)
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            s = 0
            for r = 1:size(X,2)
                s += X[i,r] * Y[j,r] * Z[k,r]
            end
            A[i,j,k] *= s
        end
    end
end
```

### Other functionality

In principle, the `@einsum` macro can flexibly implement function calls within the nested for loop structure. For example, consider transposing a block matrix:

```julia
z = Any[rand(2,2) for i=1:2, j=1:2]
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

This will work as long the function calls are outside the array names.  Again, you can use [`@macroexpand`](https://docs.julialang.org/en/stable/stdlib/base/#Base.@macroexpand) to see the exact code that is generated.



### Related Packages:

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) has less flexible syntax (and does not allow certain contractions), but can produce much more efficient code.  Instead of generating “naive” loops, it transforms the expressions into optimized contraction functions and takes care to use a good (cache-friendly) order for the looping.
