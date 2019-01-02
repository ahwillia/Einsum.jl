isdefined(Base, :__precompile__) && __precompile__()

module Einsum


using Compat # for Array{}(undef,...)

export @einsum, @einsimd, @vielsum, @vielsimd


macro einsum(ex)
    _einsum(ex) # true, false, false
end

macro einsimd(ex)
    _einsum(ex, true, true) # false
end

macro vielsum(ex)
    _einsum(ex, true, false, true)
end

macro vielsimd(ex)
    _einsum(ex, true, true, true)
end

macro einsum_checkinbounds(ex)
    _einsum(ex, false) # false, false
end


function _einsum(expr::Expr, inbounds = true, simd = false, threads = false)
    # Get left hand side (lhs) and right hand side (rhs) of equation
    lhs = expr.args[1]
    rhs = expr.args[2]

    # Get info on the left-hand side
    lhs_arrays, lhs_indices, lhs_axis_exprs = extractindices(lhs)
    length(lhs_arrays) != 1 && throw(ArgumentError(string(
        "Left-hand side of equation contains multiple arguments. Only a single ",
        "referencing expression (e.g. @einsum A[i] = ...) should be used.")))

    # Get info on the right-hand side
    rhs_arrays, rhs_indices, rhs_axis_exprs = extractindices(rhs)

    # remove duplicate indices on left-hand and right-hand side
    # and ensure that the array sizes match along these dimensions
    ###########################################################
    dimension_checks = Expr[]

    # remove duplicate indices on the right hand side
    for i in reverse(eachindex(rhs_indices))
        duplicated = false
        di = rhs_axis_exprs[i]
        
        for j = 1:(i - 1)
            if rhs_indices[j] == rhs_indices[i]
                # found a duplicate
                duplicated = true
                dj = rhs_axis_exprs[j]

                # add dimension check ensuring consistency
                push!(dimension_checks, :(@assert $dj == $di))
            end
        end
        
        for j = eachindex(lhs_indices)
            if lhs_indices[j] == rhs_indices[i]
                dj = lhs_axis_exprs[j]
                if Meta.isexpr(expr, :(:=))
                    # expr.head is :=
                    # infer the size of the lhs array
                    lhs_axis_exprs[j] = di
                else
                    # expr.head is =, +=, *=, etc.
                    lhs_axis_exprs[j] = :(min($dj, $di))
                end
                duplicated = true
            end
        end
        
        if duplicated
            deleteat!(rhs_indices, i)
            deleteat!(rhs_axis_exprs, i)
        end
    end

    # remove duplicate indices on the left hand side
    for i in reverse(eachindex(lhs_indices))
        duplicated = false
        di = lhs_axis_exprs[i]

        # don't need to check rhs, already done above

        for j = 1:(i - 1)
            if lhs_indices[j] == lhs_indices[i]
                # found a duplicate
                duplicated = true
                dj = lhs_axis_exprs[j]

                # add dimension check
                push!(dimension_checks, :(@assert $dj == $di))
            end
        end
        
        if duplicated
            deleteat!(lhs_indices, i)
            deleteat!(lhs_axis_exprs, i)
        end
    end

    # Create output array if specified by user
    @gensym T

    if Meta.isexpr(expr, :(:=))
        # infer type of allocated array
        #    e.g. rhs_arrays = [:A, :B]
        #    then the following line produces :(promote_type(eltype(A), eltype(B)))
        rhs_type = :(promote_type($([:(eltype($arr)) for arr in rhs_arrays]...)))

        type_definition = :(local $T = $rhs_type)

        output_definition = if length(lhs_axis_exprs) > 0
            :($(lhs_arrays[1]) = Array{$rhs_type}(undef, $(lhs_axis_exprs...)))
        else
            :($(lhs_arrays[1]) = zero($rhs_type))
        end

        assignment_op = :(=)
    else
        type_definition = :(local $T = eltype($(lhs_arrays[1])))
        output_definition = :(nothing)
        assignment_op = expr.head
    end

    if threads && !Meta.isexpr(expr, :(=)) && !Meta.isexpr(expr, :(:=))
        throw(ArgumentError(string(
            "Threaded @vielsum can only assign with = or := right now. ",
            "To use ", expr.head, " try @einsum instead.")))
        # could allow :(+=) by simply then removing $lhs = zero($T) line
    end

    if threads && length(lhs_indices) == 0
        # this won't actually cause problems, but won't use threads
        throw(ArgumentError(string(
            "Threaded @vielsum can't assign to a scalar LHS. ",
            "Try @einsum instead.")))
    end

    # Copy equation, expr is the Expr we'll build up and return.
    unquote_offsets!(expr)

    # Next loops to iterate over the destination variables
    if length(rhs_indices) > 0
        # There are indices on rhs that do not appear in lhs.
        # We sum over these variables.

        if !threads # then use temporaries to write into, as before

            # Innermost expression has form s += rhs
            @gensym s
            expr.args[1] = s
            expr.head = :(+=)

            # Nest loops to iterate over the summed out variables
            expr = nest_loops(expr, rhs_indices, rhs_axis_exprs, simd, false)

            # Prepend with s = 0, and append with assignment
            # to the left hand side of the equation.
            lhs_assignment = Expr(assignment_op, lhs, s)

            expr = quote
                local $s = zero($T)
                $expr
                $lhs_assignment
            end

        else # we are threading, and thus should write directly to lhs array

            expr.args[1] = lhs
            expr.head = :(+=)

            expr = nest_loops(expr, rhs_indices, rhs_axis_exprs, simd, false)

            expr = quote
                $lhs = zero($T)
                $expr
            end
        end

        # Now loop over indices appearing on lhs, if any
        expr = nest_loops(expr, lhs_indices, lhs_axis_exprs, false, threads)
    else
        # We do not sum over any indices, only loop over lhs
        expr.head = assignment_op
        expr = nest_loops(expr, lhs_indices, lhs_axis_exprs, simd, threads)
    end

    if inbounds
        expr = :(@inbounds $expr)
    end

    full_expression = quote
        $output_definition
        let
            $(dimension_checks...)
            $type_definition
            $expr
        end
        $(lhs_arrays[1])
    end

    return esc(full_expression)
end


"""
    nest_loops(expr, indices, axis_exprs, simd, threads) -> Expr

Construct a nested loop around `expr`, using `indices` in ranges `axis_exprs`.

# Example
```julia-repl
julia> nest_loops(:(A[i] = B[i]), [:i], [:(size(A, 1))], true, false)
quote
    local i
    @simd for i = 1:size(A, 1)
        A[i] = B[i]
    end
end
```
"""
function nest_loops(expr::Expr,
                    index_names::Vector{Symbol}, axis_expressions::Vector{Expr},
                    simd::Bool, threads::Bool)
    isempty(index_names) && return expr

    # Add @simd to the innermost loop, if required
    # and @threads to the outermost loop
    expr = nest_loop(expr, index_names[1], axis_expressions[1],
                     simd, threads && length(index_names) == 1)

    # Add remaining for loops
    for j = 2:length(index_names)
        expr = nest_loop(expr, index_names[j], axis_expressions[j],
                         false, threads && length(index_names) == j)
    end

    return expr
end

function nest_loop(expr::Expr, index_name::Symbol, axis_expression::Expr,
                   simd::Bool, threads::Bool)
    loop = :(for $index_name = 1:$axis_expression
                 $expr
             end)

    if threads
        loop = :(Threads.@threads $loop)
    elseif simd
        loop = :(@simd $loop)
    end

    return quote
        local $index_name
        $loop
    end
end


"""
    extractindices(expr) -> (array_names, index_names, axis_expressions)

Compute all `index_names` and respective axis calculations of an expression 
involving the arrays with `array_names`. Everything is ordered by first 
occurence in `expr`.

# Examples
```julia-repl
julia> extractindices(:(f(A[i,j,i]) + C[j]))
(Symbol[:A, :C], Symbol[:i, :j, :i, :j], Expr[:(size(A, 1)), :(size(A, 2)), :(size(A, 3)), :(size(C, 1))])
```
"""
extractindices(expr) = extractindices!(expr, Symbol[], Symbol[], Expr[])

function extractindices!(expr::Symbol,
                         array_names::Vector{Symbol},
                         index_names::Vector{Symbol},
                         axis_expressions::Vector{Expr})
    push!(array_names, expr)
    return array_names, index_names, axis_expressions
end

function extractindices!(expr::Number,
                         array_names::Vector{Symbol},
                         index_names::Vector{Symbol},
                         axis_expressions::Vector{Expr})
    return array_names, index_names, axis_expressions
end


function extractindices!(expr::Expr,
                         array_names::Vector{Symbol},
                         index_names::Vector{Symbol},
                         axis_expressions::Vector{Expr})
    if Meta.isexpr(expr, :ref) # e.g. A[i,j,k]
        array_name = expr.args[1]
        push!(array_names, array_name)

        # expr.args[2:end] are indices (e.g. [i,j,k])
        for (dimension, index_expr) in enumerate(expr.args[2:end])
            extractindex!(index_expr, array_name, dimension,
                          array_names, index_names, axis_expressions)
        end
    elseif Meta.isexpr(expr, :call)
        # e.g. 2*A[i,j], transpose(A[i,j]), or A[i] + B[j], so
        # expr.args[2:end] recursively contain the individual tensor
        # expressions (e.g. [A[i], B[j]])
        for arg in expr.args[2:end]
            extractindices!(arg, array_names, index_names, axis_expressions)
        end
    else
        throw(ArgumentError("Invalid expression head: `:$(expr.head)`"))
    end

    return return array_names, index_names, axis_expressions
end

function extractindex!(expr::Symbol, array_name::Symbol, dimension::Int,
                       array_names::Vector{Symbol},
                       index_names::Vector{Symbol},
                       axis_expressions::Vector{Expr})
    push!(index_names, expr)
    push!(axis_expressions, :(size($array_name, $dimension)))
    return array_names, index_names, axis_expressions
end

function extractindex!(expr::Number, array_name::Symbol, dimension::Int,
                       array_names::Vector{Symbol},
                       index_names::Vector{Symbol},
                       axis_expressions::Vector{Expr})
    return array_names, index_names, axis_expressions
end

function extractindex!(expr::Expr, array_name::Symbol, dimension::Int,
                       array_names::Vector{Symbol},
                       index_names::Vector{Symbol},
                       axis_expressions::Vector{Expr})
    # e.g. A[i+:offset] or A[i+5]
    #    expr is an Expr in this case
    #    We restrict it to be a Symbol (e.g. :i) followed by either
    #        a number or quoted expression.
    #    As before, push :i to index list
    #    Need to add/subtract off the offset to dimension list

    if Meta.isexpr(expr, :call) && length(expr.args) == 3
        op = expr.args[1]

        index_name = expr.args[2]
        @assert typeof(index_name) == Symbol

        offset_expr = expr.args[3]

        if offset_expr isa Integer
            offset = expr.args[3]::Integer
        elseif offset_expr isa Expr && Meta.isexpr(offset_expr, :quote)
            offset = offset_expr.args[1]
        elseif offset_expr isa QuoteNode
            offset = offset_expr.value
        else
            throw(ArgumentError("Improper expression inside reference on rhs"))
        end

        # push :i to indices we're iterating over
        push!(index_names, index_name)

        # need to invert + or - to determine iteration range
        if op == :+
            push!(axis_expressions, :(size($array_name, $dimension) - $offset))
        elseif op == :-
            push!(axis_expressions, :(size($array_name, $dimension) + $offset))
        else
            throw(ArgumentError(string("Operations inside ref on rhs are ",
                                       "limited to `+` and `-`")))
        end
    elseif Meta.isexpr(expr, :quote)
        # nothing
    else
        throw(ArgumentError("Invalid index expression: `$(expr)`"))
    end

    return array_names, index_names, axis_expressions
end


function unquote_offsets!(expr::Expr, inside_ref = false)
    inside_ref |= Meta.isexpr(expr, :ref)
    
    for i in eachindex(expr.args)
        if expr.args[i] isa Expr
            if Meta.isexpr(expr.args[i], :quote) && inside_ref # never seems to get here
                expr.args[i] = expr.args[i].args[1]
            else
                unquote_offsets!(expr.args[i], inside_ref)
            end
        end
    end

    return expr
end

# end module
############
end
