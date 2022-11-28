module Tensor_Package

using Base.Threads
using Distances
using LightGraphs
using LinearAlgebra
using PyCall
using Random
using ScikitLearn
using SparseArrays
using Statistics


include("similarity_knn.jl")
include("tensors.jl")
include("labelspreading.jl")
include("utils.jl")


end
