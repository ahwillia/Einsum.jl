#changes:
#   delete creation option -- too many corner cases
#   allow general expressions on rhs-- must recurse over expressions
#   allow quoted symbols like :a to refer to program variables
#   allow += and -= as well as = 
#   implement inbounds
#   another macro: fastcopy
#   fix cases like a[ind,ind] = x[ind] (remove dups in LHS as well as RHS)



module Einsum

export @einsum
export @einsum_inbounds
export @fastcopy
export @fastcopy_inbounds

macro einsum(eq)
	_einsum(eq, false)
end

macro einsum_inbounds(eq)
	_einsum(eq, true)
end


function _einsum(eq::Expr, inbounds::Bool)
        if eq.head == :(:(=))
            error("einsum macro no longer supports creation := option")
        end

        @assert eq.head == :(=) ||  eq.head == :(+=) || eq.head == :(-=)
        # Get left hand side (lhs) and right hand side (rhs) of eq
        @assert length(eq.args) == 2
	lhs = eq.args[1]
	rhs = eq.args[2]

	# Get info on the left-hand side
	dest_idx,dest_dim = Symbol[],Expr[]
	if typeof(lhs) == Symbol
		dest_symb = lhs
	else
		# Left hand side of equation must be a reference, e.g. A[i,j,k]
		@assert length(lhs.args) > 1
		@assert lhs.head == :ref
		dest_symb = lhs.args[1]

		# recurse expression to find indices
		lhs_rewrite = get_indices_and_rewrite!(lhs, dest_idx, dest_dim)
	end


	terms_idx,terms_dim = Symbol[],Expr[]
	rhs_rewrite = get_indices_and_rewrite!(rhs, terms_idx, terms_dim)
	terms_symb = [ terms_dim[i].args[2] for i in 1:length(terms_dim) ]
    
	# remove duplicate indices found elsewhere in terms or dest

        lhscount = length(dest_idx)
        all_idx = vcat(dest_idx, terms_idx)
        all_dim = vcat(dest_dim, terms_dim)
        firstoccurrence = Dict{Symbol, Int}()
        isfirst = zeros(Bool, length(all_idx))
        for (i,idx) in enumerate(all_idx)
            if !haskey(firstoccurrence, idx)
                firstoccurrence[idx] = i
                isfirst[i] = true
                #must be able to determine size from first occurrence
                @assert length(all_dim[i].args) > 0
            end
        end
	ex_check_dims = :()
        if !inbounds
            for (i,idx) in enumerate(all_idx)
                fi = firstoccurrence[idx]
                if fi != i && length(all_dim[i].args) > 0
                    ex_check_dims = quote
                        @assert $(esc(all_dim[fi])) == $(esc(all_dim[i]))
                        $ex_check_dims
                    end
                end
            end
        end

        # Recopy the idx and dim arrays with duplicates removed
        n = length(all_idx)
        dest_idx = all_idx[(collect(1:n) .<= lhscount)  & isfirst]
        dest_dim = all_dim[(collect(1:n) .<= lhscount)  & isfirst]
        terms_idx = all_idx[(collect(1:n) .> lhscount)  & isfirst]
        terms_dim = all_dim[(collect(1:n) .> lhscount)  & isfirst]


        ex_get_type = :($(esc(:(local T = eltype($(dest_symb))))))

	# Copy equation, ex is the Expr we'll build up and return.
	ex = Expr(eq.head, deepcopy(lhs_rewrite), deepcopy(rhs_rewrite))

	if length(terms_idx) > 0
		# There are indices on rhs that do not appear in lhs.
		# We sum over these variables.

		# Innermost expression has form s += rhs
		ex.args[1] = :s
		ex.head = :(+=)
		ex = esc(ex)

		# Nest loops to iterate over the summed out variables
		ex = nest_loops(ex,terms_idx,terms_dim)

		# Prepend with s = 0, and append with assignment
		# to the left hand side of the equation.
		ex = quote
			$(esc(:(local s = zero(T))))
			$ex 
                        $(esc(Expr(eq.head, lhs_rewrite, :s)))
		end
	else
		# We do not sum over any indices
		ex.head = eq.head
		ex = :($(esc(ex)))
	end

	# Next loops to iterate over the destination variables
	ex = nest_loops(ex,dest_idx,dest_dim)

        if inbounds
            ex = macroexpand(Expr(:macrocall, Symbol("@inbounds"), ex))
        end

	# Assemble full expression and return

	return quote
		let
		$ex_check_dims
		$ex_get_type
		$ex
                end
        end

end

function nest_loops(ex::Expr,idx::Vector{Symbol},dim::Vector{Expr})
	for (i,d) in zip(idx,dim)
		ex = quote
		    local $(esc(i)) = 1
		    for $(esc(i)) = 1:$(esc(d))
		        $(ex)
		    end
		end
	end
	return ex
end


function isvarname(s::Symbol)
    st = string(s)
    Base.isidentifier(st)
end
isvarname(s) = false



function get_indices_and_rewrite!(ex::Expr,idx_store::Vector{Symbol},dim_store::Vector{Expr})
    newexpr = Expr(ex.head)
    if ex.head == :ref 
        if isvarname(ex.args[1])
            push!(newexpr.args, ex.args[1])
        else
            push!(newexpr.args, 
                  get_indices_and_rewrite!(ex.args[1], idx_store, dim_store))
        end
        for (i, arg) in enumerate(ex.args[2:end])
            if isvarname(arg)
                push!(idx_store, arg)
                push!(dim_store, :(size($(ex.args[1]), $i)))
                push!(newexpr.args, arg)
            else
                push!(newexpr.args,
                      get_indices_and_rewrite!(arg, idx_store, dim_store))
            end
        end
    elseif ex.head == :(.) || ex.head == :call
        push!(newexpr.args, ex.args[1])
        for arg in ex.args[2:end]
            push!(newexpr.args,
                  get_indices_and_rewrite!(arg, idx_store, dim_store))
        end
    elseif ex.head == :quote
        @assert length(ex.args) == 1
        newexpr = ex.args[1]
    else
        for arg in ex.args
            push!(newexpr.args,
                  get_indices_and_rewrite!(arg, idx_store, dim_store))
        end
    end
    newexpr
end




function get_indices_and_rewrite!(sy::Symbol,idx_store::Vector{Symbol},dim_store::Vector{Expr})
    if isvarname(sy)
        push!(idx_store, sy)
        push!(dim_store, Expr(:call))  # this will cause i_has_dim in main routine to be false
    end
    sy
end

get_indices_and_rewrite!(qn::QuoteNode, ::Vector{Symbol}, ::Vector{Expr}) = qn.value

get_indices_and_rewrite!(other, ::Vector{Symbol}, ::Vector{Expr}) = other


macro fastcopy(eq)
    _fastcopy(eq, false)
end

macro fastcopy_inbounds(eq)
    _fastcopy(eq, true)
end


function _fastcopy(eq, inbounds)
    @assert eq.head == :(=)
    @assert length(eq.args) == 2
    lhs = eq.args[1]
    @assert lhs.head == :ref
    lhscoloncount = 0
    lhscolonpos = 0
    lhscolonarg = Expr(:(:))
    loopind = gensym()
    newlhs = Expr(:ref, lhs.args[1])
    for (i,ex) in enumerate(lhs.args[2:end])
        if isa(ex, Expr) && ex.head == :(:)
            lhscoloncount += 1
            lhscolonpos = i
            lhscolonarg = ex
            newex = Expr(:call, :+, ex.args[1], loopind)
        else
            newex = ex
        end
        push!(newlhs.args, newex)
    end
    @assert lhscoloncount == 1
    asserts = Any[]
    rhs = eq.args[2]
    newrhs = rewrite_rhs(rhs, loopind, asserts, lhscolonarg)
    assertq = :()
    for item in asserts
        assertq = quote
            $item
            $assertq
        end
    end
    forloopub = :( 0 : ($(lhscolonarg.args[2]) - ($(lhscolonarg.args[1]))))
    #forloopub = Expr(:(:), 0, Expr(:call, :-, lhscolonarg.args[2], lhscolonarg.args[1]))
    forloop = quote
        for $(esc(loopind)) = $(esc(forloopub))
            $(esc(newlhs)) = $(esc(newrhs))
        end
    end
    if inbounds
        forloopi = macroexpand(Expr(:macrocall, Symbol("@inbounds"), forloop))
        return forloopi
    else
        return quote
            $assertq
            $forloop
        end
    end
end

function rewrite_rhs(ex::Expr, loopind, asserts, lhscolonarg)
    if ex.head == :(:)
        newex = Expr(:call, :+, ex.args[1], loopind)
        push!(asserts,
              :(@assert length($(esc(ex))) == length($(esc(lhscolonarg)))))
    else
        newex = Expr(ex.head)
        for arg in ex.args
            push!(newex.args, rewrite_rhs(arg, loopind, asserts, lhscolonarg))
        end
    end
    newex
end

rewrite_rhs(other, loopind, asserts, lhscolonarg) = other





end


