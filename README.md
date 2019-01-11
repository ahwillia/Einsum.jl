# Einsum.jl

| **Package Build** | **Package Status** |
|:---------:|:------------------:|
| [![Build Status](https://travis-ci.org/ahwillia/Einsum.jl.svg?branch=master)](https://travis-ci.org/ahwillia/Einsum.jl) | [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) |
[![Einsum](http://pkg.julialang.org/badges/Einsum_0.6.svg)](http://pkg.julialang.org/?pkg=Einsum) | [![Project Status: Inactive - The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive) - help wanted! |


This package exports a single macro `@einsum`, which implements *similar* notation to the [Einstein
summation convention](https://en.wikipedia.org/wiki/Einstein_notation) to flexibly specify
operations on Julia `Array`s, similar to numpy's
[`einsum`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html) function
(but more flexible!).

For example, basic matrix multiplication can be implemented as:

```julia
@einsum A[i, j] := B[i, k] * C[k, j]
```

To install: `Pkg.add("Einsum")`, or else `pkg> add Einsum` after pressing `]` on Julia 0.7 and later.

## Documentation

### Basics

If the destination array is preallocated, then use `=`:

```julia
A = ones(5, 6, 7) # preallocated space
X = randn(5, 2)
Y = randn(6, 2)
Z = randn(7, 2)

# Store the result in A, overwriting as necessary:
@einsum A[i, j, k] = X[i, r] * Y[j, r] * Z[k, r]
```

If destination is not preallocated, then use `:=` to automatically create a new array for the result:

```julia
X = randn(5, 2)
Y = randn(6, 2)
Z = randn(7, 2)

# Create new array B with appropriate dimensions:
@einsum B[i, j, k] := X[i, r] * Y[j, r] * Z[k, r]
```

### What happens under the hood

To execute an expression, `@einsum` uses [Julia's metaprogramming
capabilities](http://docs.julialang.org/en/stable/manual/metaprogramming/) to generate and execute a
series of nested for loops.  To see exactly what is generated, use
[`@macroexpand`](https://docs.julialang.org/en/stable/stdlib/base/#Base.@macroexpand) (or `@expand`
from [MacroTools.jl](https://github.com/MikeInnes/MacroTools.jl)):

```julia
@macroexpand @einsum A[i, j] := B[i, k] * C[k, j]
```

The output will look much like the following (note that we "sum out" over the index `k`, since it
only occurs multiple times on the right hand side of the equation):

```julia
# determine type
T = promote_type(eltype(B), eltype(C))

# allocate new array
A = Array{T}(undef, size(B))

# check dimensions
@assert size(B, 2) == size(C, 2)

# main loop
@inbounds begin # skip bounds-checking for speed
    for i = 1:size(B, 1), j = 1:size(C, 2)
        s = zero(T)
        for k = 1:size(B,2)
            s += B[i, k] * C[k, j]
        end
        A[i, j] = s
    end
end
```

The actual generated code is a bit more verbose (and not neatly commented/formatted), and will take
care to use the right types and keep hygienic.

You can also use updating assignment operators for preallocated arrays.  E.g., `@einsum A[i, j, k] *=
X[i, r] * Y[j, r] * Z[k, r]` will produce something like

```julia
for k = 1:size(A, 3)
    for j = 1:size(A, 2)
        for i = 1:size(A, 1)
            s = 0.0
            for r = 1:size(X, 2)
                s += X[i, r] * Y[j, r] * Z[k, r]
            end
            # Difference: here, the updating form is used:
            A[i, j, k] *= s
        end
    end
end
```

### Rules for indexing variables

* Indices that show up on the right-hand-side but not the left-hand-side are summed over
* Arrays which share an index must be of the same size, in those dimensions

`@einsum` iterates over the extent of the right-hand-side indices. For example, the following code
allocates an array `A` that is the same size as `B` and copies its data into `A`:

```julia
@einsum A[i,  j] := B[i, j]  # same as A = copy(B)
```

If an index appears on the right-hand-side, but does not appear on the left-hand-side, then this
variable is summed over. For example, the following code allocates `A` to be `size(B, 1)` and sums
over the rows of `B`:

```julia
@einsum A[i] := B[i, j]  # same as A = dropdims(sum(B, dims=2), dims=2)
```

If an index variable appears multiple times on the right-hand-side, then it is asserted that the
sizes of these dimensions match. For example,

```julia
@einsum A[i] := B[i, j] * C[j]
```

will check that the second dimension of `B` matches the first dimension of `C` in length. In
particular it is equivalent to the following code:

```julia
A = zeros(size(B, 1))
@assert size(B, 2) == size(C, 1)

for i = 1:size(B, 1), j = 1:size(B, 2)
    A[i] += B[i, j] * C[j]
end
```

So an error will be thrown if the specified dimensions of `B` and `C` don't match.

#### Offset indexing 

`@einsum` also allows offsets on the right-hand-side, the range over which `i` is summed is then restricted:

```julia
@einsum A[i] = B[i - 5]
```

### `@vielsum`

This variant of `@einsum` will run multi-threaded on the outermost loop. For this to be fast, the code must not introduce temporaries like `s = 0` in the example above. Thus for example `@expand @vielsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]` results in something equivalent to `@expand`-ing the following:

```julia
Threads.@threads for k = 1:size(A,3)
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            A[i,j,k] = 0.0
            for r = 1:size(X,2)
                A[i,j,k] += X[i,r] * Y[j,r] * Z[k,r]
            end
        end
    end
end
```

For this to be useful, you will need to set an environment variable before starting Julia, such as `export JULIA_NUM_THREADS=4`. See [the manual](https://docs.julialang.org/en/latest/manual/parallel-computing/#Multi-Threading-(Experimental)-1) for details, and note that this is somewhat experimental. This will not always be faster, especially for small arrays, as there is some overhead to dividing up the work.

At present you cannot use updating assignment operators like `+=` with this macro (only `=` or `:=`) and you cannot assign to a scalar left-hand-side (only an array). These limitations prevent different threads from writing to the same memory at the same time.

### `@einsimd`

This is a variant of `@einsum` which will put `@simd` in front of the innermost loop, encouraging Julia's compiler vectorize this loop (although it may do so already). For example `@einsimd A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]` will result in approximately

```julia
@inbounds for k = 1:size(A,3)
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            s = 0.0
            @simd for r = 1:size(X,2)
                s += X[i,r] * Y[j,r] * Z[k,r]
            end
            A[i,j,k] = s
        end
    end
end
```

Whether this is a good idea or not you have to decide and benchmark for yourself in every specific case.  `@simd` makes sense for certain kinds of operations; make yourself familiar with [its documentation](https://docs.julialang.org/en/latest/base/base/#Base.SimdLoop.@simd) and the inner workings of it [in general](https://software.intel.com/en-us/articles/vectorization-in-julia).


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

This will work as long the function calls are outside the array names. Again, you can use [`@macroexpand`](https://docs.julialang.org/en/stable/stdlib/base/#Base.@macroexpand) to see the exact code that is generated.

The output need not be an array. But note that on Julia 0.7 and 1.0, the rules for evaluating in global scope (for example at the REPL prompt) are a little different -- see [this package](https://github.com/stevengj/SoftGlobalScope.jl) for instance (which is loaded in [IJulia](https://github.com/JuliaLang/IJulia.jl) notebooks). To get the same behavior as you would have inside a function, you use a `let` block:  

```julia
p = rand(5); p .= p ./ sum(p)
let
    global S
    @einsum S := - p[i] * log(p[i])
end
```

Or perhaps clearer, explicitly define a function:

```julia
m(pq, p, q) = @einsum I := pq[i,j] * log(pq[i,j] / p[i] / q[j])

m(rand(5,10), rand(5), rand(10))
```


### Related Packages:

* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) has less flexible syntax (only allowing strict Einstein convention contractions), but can produce much more efficient code.  Instead of generating “naive” loops, it transforms the expressions into optimized contraction functions and takes care to use a good (cache-friendly) order for the looping.

* [ArrayMeta.jl](https://github.com/shashi/ArrayMeta.jl) aims to produce cache-friendly operations for more general loops (but is Julia 0.6 only).
