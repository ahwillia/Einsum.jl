isdefined(Base, :__precompile__) && __precompile__()

module Einsum

using Base.Cartesian
export @einsum, @einsimd

macro einsum(ex)
    _einsum(ex)
end

macro einsimd(ex)
    _einsum(ex, true, true)
end

macro einsum_checkinbounds(ex)
    _einsum(ex, false)
end

function _einsum(ex::Expr, inbounds = true, simd = false)
    # Get left hand side (lhs) and right hand side (rhs) of equation
    lhs = ex.args[1]
    rhs = ex.args[2]

    # Get info on the left-hand side
    lhs_idx, lhs_arr, lhs_dim = extractindices(lhs)
    length(lhs_arr) != 1 && throw(ArgumentError(
        string("Left-hand side of equation contains multiple arguments. Only a single referencing",
               " expression (e.g. @einsum A[i] = ...) should be used.")))

    # Get info on the right-hand side
    rhs_idx, rhs_arr, rhs_dim = extractindices(rhs)

    # remove duplicate indices on left-hand and right-hand side
    # and ensure that the array sizes match along these dimensions
    ###########################################################
    ex_check_dims = :()

    # remove duplicate indices on the right hand side
    for i in reverse(1:length(rhs_idx))
        duplicated = false
        di = rhs_dim[i]
        for j = 1:(i - 1)
            if rhs_idx[j] == rhs_idx[i]
                # found a duplicate
                duplicated = true
                dj = rhs_dim[j]

                # add dimension check ensuring consistency
                ex_check_dims = quote
                    @assert $dj == $di
                    $ex_check_dims
                end
            end
        end
        for j = 1:length(lhs_idx)
            if lhs_idx[j] == rhs_idx[i]
                dj = lhs_dim[j]
                if ex.head == :(:=)
                    # ex.head is :=
                    # infer the size of the lhs array
                    lhs_dim[j] = di
                else
                    # ex.head is =, +=, *=, etc.
                    lhs_dim[j] = :(min($dj, $di))
                end
                duplicated = true
            end
        end
        if duplicated
            deleteat!(rhs_idx, i)
            deleteat!(rhs_dim, i)
        end
    end

    # remove duplicate indices on the left hand side
    for i in reverse(1:length(lhs_idx))
        duplicated = false
        di = lhs_dim[i]

        # don't need to check rhs, already done above

        for j = 1:(i - 1)
            if lhs_idx[j] == lhs_idx[i]
                # found a duplicate
                duplicated = true
                dj = lhs_dim[j]

                # add dimension check
                ex_check_dims = quote
                    @assert $dj == $di
                    $ex_check_dims
                end
            end
        end
        if duplicated
            deleteat!(lhs_idx, i)
            deleteat!(lhs_dim, i)
        end
    end

    # Create output array if specified by user
    ex_get_type = :(nothing)
    ex_create_arrays = :(nothing)
    ex_assignment_op = :(=)
    @gensym T

    if ex.head == :(:=)
        # infer type of allocated array
        #    e.g. rhs_arr = [:A, :B]
        #    then the following line produces :(promote_type(eltype(A), eltype(B)))
        rhs_type = :(promote_type($([:(eltype($arr)) for arr in rhs_arr]...)))

        ex_get_type = :(local $T = $rhs_type)
        
        if length(lhs_dim) > 0
            ex_create_arrays = :($(lhs_arr[1]) = Array{$rhs_type}($(lhs_dim...)))
        else
            ex_create_arrays = :($(lhs_arr[1]) = zero($rhs_type))
        end
    else
        ex_get_type = :(local $T = eltype($(lhs_arr[1])))
        ex_create_arrays = :(nothing)
        ex_assignment_op = ex.head
    end

    # Copy equation, ex is the Expr we'll build up and return.
    unquote_offsets!(ex)

    if length(rhs_idx) > 0
        # There are indices on rhs that do not appear in lhs.
        # We sum over these variables.

        # Innermost expression has form s += rhs
        @gensym s
        ex.args[1] = s
        ex.head = :(+=)

        # Nest loops to iterate over the summed out variables
        ex = nest_loops(ex, rhs_idx, rhs_dim, simd)

        lhs_assignment = Expr(ex_assignment_op, lhs, s)
        # Prepend with s = 0, and append with assignment
        # to the left hand side of the equation.
        ex = quote
            local $s = zero($T)
            $ex
            $lhs_assignment
        end
    else
        # We do not sum over any indices
        # ex.head = :(=)
        ex.head = ex_assignment_op
    end

    # Next loops to iterate over the destination variables
    ex = nest_loops(ex, lhs_idx, lhs_dim)

    if inbounds
        ex = Expr(:macrocall, Symbol("@inbounds"), ex)
    end

    full_expression = quote
        $ex_create_arrays
        let
            $ex_check_dims
            $ex_get_type
            $ex
        end
        $(lhs_arr[1])
    end

    return esc(full_expression)
end


function nest_loops(ex::Expr, idx::Vector{Symbol}, dim::Vector{Expr}, simd::Bool = false)
    isempty(idx) && return ex
    
    # Add @simd to the innermost loop, if required
    ex = nest_loop(ex, idx[1], dim[1], simd)

    # Add remaining for loops
    for j = 2:length(idx)
        ex = nest_loop(ex, idx[j], dim[j], false)
    end
    
    return ex
end

function nest_loop(ex::Expr, ix::Symbol, dim::Expr, simd::Bool)
    loop = :(for $ix = 1:$dim
                 $ex
             end)
    
    if simd
        loop = Expr(:macrocall, Symbol("@simd"), loop)
    end
    
    return quote
        local $ix
        $loop
    end
end


extractindices(ex) = extractindices!(ex, Symbol[], Symbol[], Expr[])

function extractindices!(ex::Symbol,
                         idx_store::Vector{Symbol},
                         arr_store::Vector{Symbol},
                         dim_store::Vector{Expr})
    push!(arr_store, ex)
    return idx_store, arr_store, dim_store
end

function extractindices!(ex::Number,
                         idx_store::Vector{Symbol},
                         arr_store::Vector{Symbol},
                         dim_store::Vector{Expr})
    return idx_store, arr_store, dim_store
end


function extractindices!(ex::Expr,
                         idx_store::Vector{Symbol},
                         arr_store::Vector{Symbol},
                         dim_store::Vector{Expr})
    if ex.head == :ref # e.g. A[i,j,k]
        arrname = ex.args[1]
        push!(arr_store, arrname)
        
        # ex.args[2:end] are indices (e.g. [i,j,k])
        for (pos, idx) in enumerate(ex.args[2:end])
            extractindex!(idx, arrname, pos, idx_store, arr_store, dim_store)
        end
    elseif ex.head == :call
        # e.g. 2*A[i,j], transpose(A[i,j]), or A[i] + B[j], so
        # ex.args[2:end] are the individual tensor expressions (e.g. [A[i], B[j]])
        for arg in ex.args[2:end]
            extractindices!(arg, idx_store, arr_store, dim_store)
        end
    else
        throw(ArgumentError("Invalid expression head: `:$(ex.head)`"))
    end
    
    return idx_store, arr_store, dim_store
end

function extractindex!(ex::Symbol, arrname, position,
                       idx_store, arr_store, dim_store)
    push!(idx_store, ex)
    push!(dim_store, :(size($arrname, $position)))
    return idx_store, arr_store, dim_store
end

# TODO: Union{Number, QuoteNode}?
function extractindex!(ex::Number, arrname, position,
                       idx_store, arr_store, dim_store)
    return idx_store, arr_store, dim_store
end

function extractindex!(ex::Expr, arrname, position,
                       idx_store, arr_store, dim_store)
    # e.g. A[i+:offset] or A[i+5]
    #    ex is an Expr in this case
    #    We restrict it to be a Symbol (e.g. :i) followed by either
    #        a number or quoted expression.
    #    As before, push :i to index list
    #    Need to add/subtract off the offset to dimension list
    
    if ex.head == :call && length(ex.args) == 3
        op = ex.args[1]
        
        idx = ex.args[2]
        @assert typeof(idx) == Symbol
        
        off_expr = ex.args[3]
        
        if off_expr isa Integer
            off = ex.args[3]::Integer
        elseif off_expr isa Expr && off_expr.head == :quote
            off = off_expr.args[1]
        elseif off_expr isa QuoteNode
            off = off_expr.value
        else
            throw(ArgumentError("Improper expression inside reference on rhs"))
        end
        
        # push :i to indices we're iterating over
        push!(idx_store, idx)
        
        # need to invert + or - to determine iteration range
        if op == :+
            push!(dim_store, :(size($arrname, $position) - $off))
        elseif op == :-
            push!(dim_store, :(size($arrname, $position) + $off))
        else
            throw(ArgumentError("Operations inside ref on rhs are limited to `+` or `-`"))
        end
    elseif ex.head == :quote
        # nothing
    else
        throw(ArgumentError("Invalid index expression: `$(ex)`"))
    end

    return idx_store, arr_store, dim_store
end


function unquote_offsets!(ex::Expr, inside_ref = false)
    inside_ref |= Meta.isexpr(ex, :ref)
    
    for i in eachindex(ex.args)
        if ex.args[i] isa Expr
            if Meta.isexpr(ex.args[i], :quote) && inside_ref
                ex.args[i] = ex.args[i].args[1]
            else
                unquote_offsets!(ex.args[i], inside_ref)
            end
        end
    end
    
    return ex
end

# end module
############
end
