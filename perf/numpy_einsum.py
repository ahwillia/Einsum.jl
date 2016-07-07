import numpy as np
import timeit

def benchmark_numpy(dim=30):
	A = np.random.randn(dim,dim)
	B = np.random.randn(dim,dim)
	C = np.random.randn(dim,dim)
	D = np.random.randn(dim,dim)
	E = np.random.randn(dim,dim)

	return np.einsum('ra,rb,rc,rd,re->abcde',A,B,C,D,E);

times = timeit.Timer(benchmark_numpy).timeit(number=1)

print times
