module Einsum

export @einsum

macro einsum(eq)
	_einsum(eq)
end

function _einsum(eq::Expr)
	
	# Get left hand side (lhs) and right hand side (rhs) of eq
	@assert eq.head == :(=)
	lhs = eq.args[1]
	rhs = eq.args[2]

	# Left hand side of equation must be a reference, e.g. A[i,j,k]
	@assert length(lhs.args) > 1
	@assert lhs.head == :ref

	# recurse expression to find indices
	dest_idx,dest_dim = Symbol[],Expr[]
	get_indices!(lhs,dest_idx,dest_dim)

	terms_idx,terms_dim = Symbol[],Expr[]
	get_indices!(rhs,terms_idx,terms_dim)

	# remove duplicate indices found elsewhere in terms or dest
	ex_check_dims = :()
	for i in reverse(1:length(terms_idx))
		duplicated = false
		di = terms_dim[i]
		for j = 1:(i-1)
			if terms_idx[j] == terms_idx[i]
				dj = terms_dim[j]
				ex_check_dims = quote
					@assert $(esc(dj)) == $(esc(di))
					$ex_check_dims
				end
				duplicated = true
			end
		end
		for j = 1:length(dest_idx)
			if dest_idx[j] == terms_idx[i]
				dj = dest_dim[j]
				ex_check_dims = quote
					@assert $(esc(dj)) == $(esc(di))
					$ex_check_dims
				end
				duplicated = true
			end
		end
		if duplicated
			deleteat!(terms_idx,i)
			deleteat!(terms_dim,i)
		end
		i -= 1
	end

	# Copy equation, ex is the Expr we'll build up and return.
	ex = deepcopy(eq)

	# Innermost expression has form s += rhs
	ex.args[1] = :s
	ex.head = :(+=)
	ex = esc(ex)

	ex = nest_loops(ex,terms_idx,terms_dim)

	ex = quote
		$(esc(:(local s = 0)))
		$ex 
		$(esc(:($lhs = s)))
	end

	ex = nest_loops(ex,dest_idx,dest_dim)

	return quote
	$ex_check_dims
	$ex
	end
end

function nest_loops(ex::Expr,idx::Vector{Symbol},dim::Vector{Expr})
	for (i,d) in zip(idx,dim)
		ex = quote
		    for $(esc(i)) = 1:$(esc(d))
		        $(ex)
		    end
		end
	end
	return ex
end


function get_indices!(ex::Expr,idx_store::Vector{Symbol},dim_store::Vector{Expr})
	if ex.head == :ref
		for (i,arg) in enumerate(ex.args[2:end])
			push!(idx_store,arg)
			push!(dim_store,:(size($(ex.args[1]),$i)))
		end
	else
		@assert ex.head == :call
		for arg in ex.args[2:end]
			get_indices!(arg,idx_store,dim_store)
		end
	end
end

end
