import Einsum
import Benchmarks

let
	"""
	Common fate bechmark proposed by @faroit
	"""
	function commonfate(A,H,C)
	  Einsum.@einsum P[a,b,f,t,c] := A[a,b,f,j]*H[t,j]*C[c,j]
	  return nothing
	end

	function assess_benchmark(dim=30, r=10)
		A = randn(dim,dim,dim,r)
	    H = randn(dim,r)
	    C = randn(dim,r)

	    result = Benchmarks.@benchmark commonfate(A,H,C) 
	    stat = Benchmarks.SummaryStatistics(result)

	    # convert ns -> seconds
	    t = stat.elapsed_time_center / 1_000_000_000.0

	    return (dim,t)
	end

	assess_benchmark()
end
