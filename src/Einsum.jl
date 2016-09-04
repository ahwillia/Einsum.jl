isdefined(Base, :__precompile__) && __precompile__()

module Einsum

export @einsum, @einsimd

macro einsum(eq)
    _einsum(eq)
end

macro einsimd(eq)
    _einsum(eq,true,true)
end

macro einsum_checkinbounds(eq)
    _einsum(eq,false)
end

function _einsum(eq::Expr, inbound=true, simd=false)
    
    # Get left hand side (lhs) and right hand side (rhs) of eq
    lhs = eq.args[1]
    rhs = eq.args[2]

    # Get info on the left-hand side
    lhs_idx,lhs_arr,lhs_dim = get_indices!(lhs)
    @assert length(lhs_arr) == 1

    # Get info on the right-hand side
    rhs_idx,rhs_arr,rhs_dim = get_indices!(rhs)

    # remove duplicate indices found elsewhere in terms or dest
    ex_check_dims = :()
    for i in reverse(1:length(rhs_idx))
        duplicated = false
        di = rhs_dim[i]
        for j = 1:(i-1)
            if rhs_idx[j] == rhs_idx[i]
                dj = rhs_dim[j]
                ex_check_dims = quote
                    @assert $(esc(dj)) == $(esc(di))
                    $ex_check_dims
                end
                duplicated = true
            end
        end
        for j = 1:length(lhs_idx)
            if lhs_idx[j] == rhs_idx[i]
                dj = lhs_dim[j]
                if eq.head == :(:=)
                    lhs_dim[j] = di 
                else
                    # eq.head is =, +=, *=, etc.
                    lhs_dim[j] = :(min($dj,$di))
                end 
                duplicated = true
            end
        end
        if duplicated
            deleteat!(rhs_idx,i)
            deleteat!(rhs_dim,i)
        end
        i -= 1
    end

    # Create output array if specified by user 
    ex_get_type = :(nothing)
    ex_create_arrays = :(nothing)
    ex_assignment_op = :(=)
    
    if eq.head == :(:=)
        ex_get_type = :($(esc(:(local T = eltype($(rhs_arr[1]))))))
        if length(lhs_dim) > 0
            ex_create_arrays = :($(esc(:($(lhs_arr[1]) = Array(eltype($(rhs_arr[1])),$(lhs_dim...))))))
        else
            ex_create_arrays = :($(esc(:($(lhs_arr[1]) = zero(eltype($(rhs_arr[1])))))))
        end
    else
        ex_get_type = :($(esc(:(local T = eltype($(lhs_arr[1]))))))
        ex_create_arrays = :(nothing)
        ex_assignment_op = eq.head
    end 

    # Copy equation, ex is the Expr we'll build up and return.
    ex = deepcopy(eq)

    if length(rhs_idx) > 0
        # There are indices on rhs that do not appear in lhs.
        # We sum over these variables.

        # Innermost expression has form s += rhs
        ex.args[1] = :s
        ex.head = :(+=)
        ex = esc(ex)

        # Nest loops to iterate over the summed out variables
        ex = nest_loops(ex,rhs_idx,rhs_dim,simd)


        lhs_assignment = Expr(ex_assignment_op, lhs, :s)
        # Prepend with s = 0, and append with assignment
        # to the left hand side of the equation.
        ex = quote
            $(esc(:(local s = zero(T))))
            $ex 
            $(esc(lhs_assignment))
        end
    else
        # We do not sum over any indices
        # ex.head = :(=)
        ex.head = ex_assignment_op
        ex = :($(esc(ex)))
    end

    # Next loops to iterate over the destination variables
    ex = nest_loops(ex,lhs_idx,lhs_dim)

    # Assemble full expression and return
    return quote
        $ex_create_arrays
        let
            @inbounds begin
                $ex_check_dims
                $ex_get_type
                $ex
            end
        end
    end
end

function nest_loops(ex::Expr,idx::Vector{Symbol},dim::Vector{Expr},simd=false)
    if simd && !isempty(idx)
        # innermost index and dimension
        i = idx[1]
        d = dim[1]

        # Add @simd to the innermost loop.
        ex = quote
            local $(esc(i))
            @simd for $(esc(i)) = 1:$(esc(d))
                $(ex)
            end
        end
        start_ = 2
    else
        start_ = 1
    end

    # Add remaining for loops
    for j = start_:length(idx)
        # index and dimension we are looping over
        i = idx[j]
        d = dim[j]

        # add for loop around expression
        ex = quote
            local $(esc(i))
            for $(esc(i)) = 1:$(esc(d))
                $(ex)
            end
        end
    end
    return ex
end

function get_indices!(
        ex::Symbol,
        idx_store=Symbol[],
        arr_store=Symbol[ex],
        dim_store=Expr[]
    )
    return idx_store,arr_store,dim_store
end

function get_indices!(
        ex::Expr,
        idx_store=Symbol[],
        arr_store=Symbol[],
        dim_store=Expr[]
    )

    if ex.head == :ref
        # e.g. A[i,j,k] #
        push!(arr_store, ex.args[1])

        # iterate over indices (e.g. i,j,k)
        for (i,arg) in enumerate(ex.args[2:end])
            
            if typeof(arg) == Symbol
                # e.g. A[i]
                #    First, push :i to index list
                #    Second, push size(A,1) to dimension list
                push!(idx_store,arg)
                push!(dim_store,:(size($(ex.args[1]),$i)))
            
            elseif typeof(arg) == QuoteNode
                # e.g. A[:constant]
                #    Do nothing, since we don't iterate over this dimension
                continue
            else
                # e.g. A[i+:offset] or A[i+5]
                #    arg is an Expr in this case
                #    We restrict it to be a Symbol (e.g. :i) followed by either
                #        a number or quoted expression.
                #    As before, push :i to index list
                #    Need to add/subtract off the offset to dimension list
                @assert arg.head == :call
                op = arg.args[1]
                sym = arg.args[2]
                offT = typeof(arg.args[3])
                if offT == QuoteNode
                    off = arg.args[3].value::Symbol
                elseif offT <: Integer
                    off = arg.args[3]::Integer
                else
                    throw(ArgumentError("improper expression inside reference on rhs"))
                end
                @assert typeof(sym) == Symbol

                # push :i to indices we're iterating over
                push!(idx_store, sym)

                # need to invert + or - to determine iteration range
                if op == :+
                    push!(dim_store,:( (size($(ex.args[1]),$i) - $off )))
                elseif op == :-
                    push!(dim_store,:( (size($(ex.args[1]),$i) + $off )))
                else
                    throw(ArgumentError("operations inside ref on rhs are limited to + or -"))
                end
            end
        end
    else
        # e.g. 2*A[i,j] or transpose(A[i,j])
        @assert ex.head == :call
        for arg in ex.args[2:end]
            get_indices!(arg,idx_store,arr_store,dim_store)
        end
    end
    idx_store,arr_store,dim_store
end

end # module
