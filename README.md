[![Build Status](https://travis-ci.org/ahwillia/Einsum.jl.svg?branch=master)](https://travis-ci.org/ahwillia/Einsum.jl)
[![Einsum](http://pkg.julialang.org/badges/Einsum_0.4.svg)](http://pkg.julialang.org/?pkg=Einsum)

# Einsum.jl
Einstein summation notation in Julia. Similar to numpy's [`einsum`](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html) function. To install: `Pkg.add("Einsum")`.


### Usage of @einsum:


```julia
using Einsum
A = zeros(5,6,7); # need to preallocate destination
X = randn(5,2)
Y = randn(6,2)
Z = randn(7,2)
@einsum A[i,j,k] = X[i,r]*Y[j,r]*Z[k,r]
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

### Rationale

Many of the operations possible with `@einsum` can also be
expressed using Julia's colon notation or by working with
entire arrays. Julia array operations 
almost always create temporaries.  Especially
in the case of working with short vectors in an inner
loop, the creation of temporaries can kill performance.
Thus, for-loops may be preferable, and `@einsum` provides
a means to express for-loops succinctly.

A particular application of `@einsum` is to finite element
analysis.  Often the inner assembly loop involves many
operations on short vectors.


### Calling sequences; other functionality

The LHS must be a subscripted
array expression.  The RHS can be an arbitrary expression.  All
variable names on both sides of the equation are interpreted as
dummy indices.  The exceptions are as follows: a variable name
followed by square brackets (to indicate subscripting), or followed
by parentheses (to indicate function call), or followed by a dot
(to indicate field membership) are treated literally rather
than as dummies.

If a program variable is needed literally in the LHS or RHS and does
not fall into the three categories mentioned in the
last paragraph, precede it with
a colon to indicate that it is not a dummy variable.

```julia
@einsum A[i,:j] = X[i,:j] + y[i]
```

In the above snippet, `i` is a dummy variable while `j` is a program
variable.

The assignment operation  may also be `+=` or `-=`. 

A variant of `@einsub` is `@einsub_inbounds`, which skips
over consistent size-checks and 
bounds checking in the subscripting operation.

Again, you can use [`macroexpand`](http://docs.julialang.org/en/release-0.4/manual/metaprogramming/#macros) to see the exact code that is generated.

### Usage of `@fastcopy` 

The `@fastcopy` macro carries out array copying operations 
indicated by a colon subscript using
a loop instead of Julia's array operations.  Again, the
purpose is to avoid creation of temporaries.  Here is an
example:

```julia
@fastcopy  a[m:n] = b[m:n] + c[r:p]
```
For this to be valid one must have `p-r==n-m`. 

The `@fastcopy` macro may be used in tandem with the
`@einsum` macro in applications
of the following form.  One starts with a big
vector, e.g., the assembled load vector in finite element analysis.
From this, one
copies out a small vector using `@fastcopy`, operates on it with
`@einsum`, and then replaces it in the big vector with `@fastcopy`.

An alternative would be the `sub` operation to extract subarrays
of the big vector, but this operation has the drawback that `sub` returns
a heap-allocated object.  This can be a performance-killer in an 
inner loop that operates on short vectors.  In this setting,
copying in and out with `@fastcopy` may be preferable.

The `@fastcopy_inbounds` macro performs the same function
as `@fastcopy` but skips all bounds checking.


### Benchmarks:

See the `benchmarks/` folder for code.

Julia:

```julia
julia> @time benchmark_einsum(30)
  2.237183 seconds (12 allocations: 185.429 MB, 0.23% gc time)
```

Python:

```python
In [2]: %timeit benchmark_numpy(30)
1 loop, best of 3: 9.27 s per loop
```

### Related Packages:

* https://github.com/Jutho/TensorOperations.jl
