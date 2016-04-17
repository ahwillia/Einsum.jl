module Einsum

export @einsum

macro einsum(eq)
	_einsum(eq)
end

function _einsum(eq::Expr)
	
	@assert eq.head == :(=)

	lhs = eq.args[1] # left hand side of equation
	rhs = eq.args[2] # right hand side of equation

	@assert length(lhs.args) > 1
	@assert lhs.head == :ref

	dest_idx,dest_dim = Symbol[],Expr[]
	get_indices!(lhs,dest_idx,dest_dim)

	terms_idx,terms_dim = Symbol[],Expr[]
	get_indices!(rhs,terms_idx,terms_dim)

	# remove duplicate indices found elsewhere in terms or dest 
	i = length(terms_idx)
	while i > 0
		if terms_idx[i] in terms_idx[1:(i-1)] || terms_idx[i] in dest_idx
			deleteat!(terms_idx,i)
			deleteat!(terms_dim,i)
		end
		i -= 1
	end
	
	# stick ex into middle of bunch of nested loops

	lhs_eq_s = :($lhs = s) 
	ex = deepcopy(eq)
	ex.args[1] = :s
	ex.head = :(+=)
	ex = esc(ex)

	ex = nest_loops(ex,terms_idx,terms_dim)

	ex = quote
		$(esc(:(local s = 0)))
		$ex 
		$(esc(lhs_eq_s))
	end

	ex = nest_loops(ex,dest_idx,dest_dim)

	return ex
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
