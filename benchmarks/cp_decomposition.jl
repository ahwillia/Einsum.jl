import Einsum
import Benchmarks

let

	"""
	Composes a 5th-order tensor from 5 factors. This is a
	CP decomposition, PARAFAC, or CANDECOMP decomposition of
	tensors.
	"""
	function cp5(A,B,C,D,E)
	    Einsum.@einsum X[a,b,c,d,e] := A[r,a]*B[r,b]*C[r,c]*D[r,d]*E[r,e]
	    return X
	end

	function assess_benchmark(dim=5000, r=10)
		A = randn(dim,r)
	    B = randn(dim,r)
	    C = randn(dim,r)
	    D = randn(dim,r)
	    E = randn(dim,r)

	    result = Benchmarks.@benchmark cp5(A,B,C,D,E)
	    stat = Benchmarks.SummaryStatistics(result)

	    # convert ns -> seconds
	    t = stat.elapsed_time_center / 1_000_000_000.0

	    return (dim,t)
	end

	assess_benchmark()
end
