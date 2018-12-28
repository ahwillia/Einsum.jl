# Einsum.jl
Einstein summation notation similar to numpy's [`einsum`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html) function (but more flexible!).

| **PackageEvaluator** | **Package Build** | **Package Status** |
|:--------------------:|:---------:|:------------------:|
| [![Einsum](http://pkg.julialang.org/badges/Einsum_0.7.svg)](http://pkg.julialang.org/?pkg=Einsum) | [![Build Status](https://travis-ci.org/ahwillia/Einsum.jl.svg?branch=master)](https://travis-ci.org/ahwillia/Einsum.jl) | [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) |
[![Einsum](http://pkg.julialang.org/badges/Einsum_0.6.svg)](http://pkg.julialang.org/?pkg=Einsum) | | [![Project Status: Inactive - The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive) - help wanted! |

To install: `Pkg.add("Einsum")`, or else `pkg> add Einsum` after pressing `]` on Julia 0.7 and later.

## Documentation

### Basics

If the destination array is preallocated, then use `=`:

```julia
using Einsum
A = ones(5,6,7) # will be overwritten
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
```

If destination is not preallocated, then use `:=` to automatically create a new array `B` with appropriate dimensions:

```julia
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum B[i,j,k] := X[i,r]*Y[j,r]*Z[k,r]
```

### What happens under the hood

To see exactly what is generated, use [`@macroexpand`](https://docs.julialang.org/en/stable/stdlib/base/#Base.@macroexpand) (or `@expand` from [MacroTools.jl](https://github.com/MikeInnes/MacroTools.jl):

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

In reality, this code will be preceded by allocations if necessary, and size checks. It will be wrapped in `@inbounds` to disable bounds checking during the loops. And it will take care to use the right types, and keep hygenic.

You can also use updating assignment operators for preallocated arrays.  E.g., `@einsum A[i,j,k] *= X[i,r]*Y[j,r]*Z[k,r]` will produce something like

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

### `@vielsum`

This variant of `@einsum` will run multi-threaded on the outermost loop. For this to be fast, the code must not introduce temporaries like `s = 0` in the example above. Thus for example `@expand @vielsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]` results in something equivalent to `@expand`-ing the following:

```julia
Threads.@threads for k = 1:size(A,3)
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            A[i,j,k] = 0
            for r = 1:size(X,2)
                A[i,j,k] += X[i,r] * Y[j,r] * Z[k,r]
            end
        end
    end
end
```

For this to be useful, you will need to set an environment variable before starting Julia, such as `export JULIA_NUM_THREADS=4`. See [the manual](https://docs.julialang.org/en/stable/manual/parallel-computing/#Multi-Threading-(Experimental)-1) for details, and note that this is somewhat experimental. This will not always be faster, especially for small arrays, as there is some overhead to dividing up the work.

At present you cannot use updating assignment operators like `+=` with this macro, only `=` or `:=`. And you cannot assign to a scalar left-hand-side, only an array.

### `@einsimd`

This is a variant of `@einsum` which will put `@simd` in front of the innermost loop; e.g., `@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]` will result approximately in

```julia
for k = 1:size(A,3)
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            s = 0
            @simd for r = 1:size(X,2)
                s += X[i,r] * Y[j,r] * Z[k,r]
            end
            A[i,j,k] = s
        end
    end
end
```

Whether this is a good idea or not you have to decide and benchmark for yourself in every specific case.  `@simd` makes sense for certain kinds of operations; make yourself familiar with [its documentation](https://docs.julialang.org/en/stable/manual/performance-tips/#Performance-Annotations-1) and the inner workings of it [in general](https://software.intel.com/en-us/articles/vectorization-in-julia).


### Other functionality

The `@einsum` macro can implement function calls within the nested for loop structure. For example, consider transposing a block matrix:

```julia
z = Any[rand(2,2) for i=1:2, j=1:2]
@einsum t[i,j] := transpose(z[j,i])
```

This produces a for loop structure with a `transpose` function call in the middle. Approximately:

```julia
for j = 1:size(z,1)
    for i = 1:size(z,2)
        t[i,j] = transpose(z[j,i])
    end
end
```

This will work as long the function calls are outside the array names.

The output need not be an array. But note that on Julia 0.7 and 1.0, the rules for evaluating in global scope (for example at the REPL prompt) are a little different -- see [this package](https://github.com/stevengj/SoftGlobalScope.jl) for instance. To get the same behavior as you would have inside a function, you can do this:  

```julia
let
    global S
    @einsum S := - p[i] * log(p[i])
end
```

Again, you can use [`@macroexpand`](https://docs.julialang.org/en/stable/stdlib/base/#Base.@macroexpand) to see the exact code that is generated.


### Related Packages:

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) has less flexible syntax (only allowing strict Einstein convention contractions), but can produce much more efficient code.  Instead of generating “naive” loops, it transforms the expressions into optimized contraction functions and takes care to use a good (cache-friendly) order for the looping.

* [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) aims to produce cache-friendly operations for more general loops (but is Julia 0.6 only).
