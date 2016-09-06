isdefined(Base, :__precompile__) && __precompile__()

module Einsum

using Base.Cartesian
export @einsum, @einsimd

macro einsum(ex)
    _einsum(ex)
end

macro einsimd(ex)
    _einsum(ex,true,true)
end

macro einsum_checkinbounds(ex)
    _einsum(ex,false)
end

function _einsum(ex::Expr, inbound=true, simd=false)
    
    # Get left hand side (lhs) and right hand side (rhs) of equation
    lhs = ex.args[1]
    rhs = ex.args[2]

    # Get info on the left-hand side
    lhs_idx,lhs_arr,lhs_range = get_indices!(lhs;lhs=true)
    @assert length(lhs_arr) == 1

    # Get info on the right-hand side
    rhs_idx,rhs_arr,rhs_range = get_indices!(rhs;lhs=false)

    # for each loop index find the maximal range such that all array
    # ref-expressions are inbound by intersecting the respective ranges
    
    # lookup range by index variable symbol
    range_by_idx = Dict{Symbol,Expr}()
    
    # process RHS indices
    for jj=1:length(rhs_idx)
      idx = rhs_idx[jj]
      if idx ∈ keys(range_by_idx)
        range_by_idx[idx] = intersect_ranges(range_by_idx[idx], rhs_range[jj])
      else
        range_by_idx[idx] = rhs_range[jj]
      end
    end

    if ex.head != :(:=)
      # process LHS indices
      for jj=1:length(lhs_idx)
        idx = lhs_idx[jj]
        if idx ∈ keys(range_by_idx)
          range_by_idx[idx] = intersect_ranges(range_by_idx[idx], lhs_range[jj])
        else
          range_by_idx[idx] = lhs_range[jj]
        end
      end
    end
    
    # remove duplicate indices found elsewhere in terms or dest
    # ex_check_dims = :()
    for i in reverse(1:length(rhs_idx))
        duplicated = false
        # di = rhs_dim[i]
        for j = 1:(i-1)
            if rhs_idx[j] == rhs_idx[i]
                # dj = rhs_dim[j]
                # ex_check_dims = quote
                #     @assert $(esc(dj)) == $(esc(di))
                #     $ex_check_dims
                # end
                duplicated = true
            end
        end
        for j = 1:length(lhs_idx)
            if lhs_idx[j] == rhs_idx[i]
                # dj = lhs_dim[j]
                # if ex.head == :(:=)
                #     lhs_dim[j] = di 
                # else
                #     # ex.head is =, +=, *=, etc.
                #     lhs_dim[j] = :(min($dj,$di))
                # end 
                duplicated = true
            end
        end
        if duplicated
            deleteat!(rhs_idx,i)
            # deleteat!(rhs_dim,i)
        end
        i -= 1
    end

    # Create output array if specified by user 
    ex_get_type = :(nothing)
    ex_create_arrays = :(nothing)
    ex_assignment_op = :(=)
    
    if ex.head == :(:=)
        
        # infer type of allocated array
        #    e.g. rhs_arr = [:A,:B]
        #    then the following line produces :(promote_type(eltype(A),eltype(B)))
        rhs_type = Expr(:call,:promote_type, [ Expr(:call,:eltype,arr) for arr in rhs_arr ]...)
        lhs_sizes=[:(length($(range_by_idx[idx]))) for idx=lhs_idx]
        
        ex_get_type = :($(esc(:(local T = $rhs_type))))
        if length(lhs_sizes) > 0
            ex_create_arrays = :($(esc(:($(lhs_arr[1]) = Array($rhs_type,$(lhs_sizes...))))))
        else
            ex_create_arrays = :($(esc(:($(lhs_arr[1]) = zero($rhs_type)))))
        end
    else
        ex_get_type = :($(esc(:(local T = eltype($(lhs_arr[1]))))))
        ex_create_arrays = :(nothing)
        ex_assignment_op = ex.head
    end 

    # Copy equation, ex is the Expr we'll build up and return.
    remove_quote_nodes!(ex)

    if length(rhs_idx) > 0
        # There are indices on rhs that do not appear in lhs.
        # We sum over these variables.

        # Innermost expression has form s += rhs
        ex.args[1] = :s
        ex.head = :(+=)
        ex = esc(ex)

        # Nest loops to iterate over the summed out variables
        ex = nest_loops(ex,rhs_idx,range_by_idx,simd)

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
    ex = nest_loops(ex,lhs_idx,range_by_idx)

    # Assemble full expression and return
    return quote
        $ex_create_arrays
        let
            @inbounds begin
                # $ex_check_dims
                $ex_get_type
                $ex
            end
        end
    end
end


function intersect_ranges(rng1::Expr, rng2::Expr)
  @assert rng1.head == :(:)
  @assert rng2.head == :(:)
  
  if rng1.args[1] == rng2.args[1]
    lo = rng1.args[1]
  else
    lo = :(max($(rng1.args[1]),$(rng2.args[1])))
  end
  
  if rng1.args[2] == rng2.args[2]
    hi = rng1.args[2]
  else
    hi = :(min($(rng1.args[2]),$(rng2.args[2])))
  end
  :($(lo):$(hi))
end

function nest_loops(ex::Expr,idx::Vector{Symbol},rng_by_idx::Dict{Symbol,Expr},simd=false)
    if simd && !isempty(idx)
        # innermost index and dimension
        i = idx[1]

        # Add @simd to the innermost loop.
        ex = quote
            local $(esc(i))
            @simd for $(esc(i)) = $(esc(rng_by_idx[i]))
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

        # add for loop around expression
        ex = quote
            local $(esc(i))
            for $(esc(i)) = $(esc(rng_by_idx[i]))
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
        rng_store=Expr[];
        lhs=true
    )
    return idx_store,arr_store,rng_store
end

@inline get_indices!(ex::Number,args...;lhs=true) = (args...)

function get_indices!(
        ex::Expr,
        idx_store=Symbol[],
        arr_store=Symbol[],
        rng_store=Expr[];
        lhs=true
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
                push!(rng_store,:(1:size($(ex.args[1]),$i)))
            
            elseif typeof(arg) <: Number
                # e.g. A[5]
                #    Do nothing, since we don't iterate over this dimension
                continue
            else
                # e.g. A[i+:offset] or A[i+5]
                #    arg is an Expr in this case
                #    We restrict it to be a Symbol (e.g. :i) followed by either
                #        a number or quoted expression.
                #    As before, push :i to index list
                #    Need to add/subtract off the offset to dimension list
                arg.head == :quote && continue
                
                if lhs
                    error("Cannot have offsets on LHS: $(arg)")
                end
                @assert arg.head == :call
                @assert length(arg.args) == 3
                op = arg.args[1]
                sym = arg.args[2]
                offT = typeof(arg.args[3])
                if offT <: Integer
                    off = arg.args[3]::Integer
                elseif offT <: Expr && arg.args[3].head == :quote
                    off = arg.args[3].args[1]::Symbol
                elseif offT == QuoteNode
                    off = arg.args[3].value::Symbol
                else
                    throw(ArgumentError("improper expression inside reference on rhs"))
                end
                @assert typeof(sym) == Symbol

                # push :i to indices we're iterating over
                push!(idx_store, sym)

                # need to invert + or - to determine iteration range
                if op == :+
                    push!(rng_store,:(1+max(0,-$off):(size($(ex.args[1]),$i)+min(0,-$off ))))
                elseif op == :-
                    push!(rng_store,:(1+max(0,$off):(size($(ex.args[1]),$i)+min(0,$off ))))
                else
                    throw(ArgumentError("operations inside ref on rhs are limited to + or -"))
                end
              
            end
        end
    else
        # e.g. 2*A[i,j] or transpose(A[i,j])
        @assert ex.head == :call
        for arg in ex.args[2:end]
            get_indices!(arg,idx_store,arr_store,rng_store; lhs=lhs)
        end
    end
    idx_store,arr_store,rng_store
end

function remove_quote_nodes!(ex::Expr)
    for i = 1:length(ex.args)
        if typeof(ex.args[i]) == Expr
            if ex.args[i].head == :quote
                ex.args[i] = :($(ex.args[i].args[1]))
            else
                remove_quote_nodes!(ex.args[i])
            end
        end
    end
    return ex
end

end # module
