
using PyCall

py"""exec(open("train.py").read())"""

@pyimport scipy.sparse as sp

sp.csr_matrix(a)

push!(pyimport("sys")["path"], pwd())

using SparseArrays
a = spzeros(10, 10)

a[1, 2] = 5

a
PyObject(a)

pyimport("train_JL")[:train](sp.csr_matrix(a), nothing, nothing, nothing, nothing, nothing, nothing)
