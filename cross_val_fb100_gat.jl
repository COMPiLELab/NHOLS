include("Tensor_Package/Tensor_Package.jl")

using .Tensor_Package.Tensor_Package
using Base.Threads
using BSON
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays

include("CV_helpers.jl")
include("graph_sage.jl")

function read_hols_datasets(dataset)
    IJ = readdlm("./data/fb100/edges-$dataset.txt")
    Y = readdlm("./data/fb100/labels-$dataset.txt")
    n = length(Y)
    A = sparse(IJ[:,1], IJ[:,2], 1, n, n)
    A = max.(A, A')
    A = min.(A, 1)
    return A, convert(Vector{Int64}, vec(Y))
end

using BSON, SparseArrays
data = BSON.load("submission_material\\NHOLS-data/train_val_inds_optdigits.bson")
data







function GAT(dataset_name):
    data = BSON.load("submission_material\\NHOLS-data/train_val_inds_$dataset_name.bson")
    d
#data["all_train_inds"][0.05], data["all_val_inds"][0.05]

function main(which)
    Random.seed!(1234)

    

    # higher-order mixing functions
    f_geom(x,y) = 2.0 * sqrt(x * y)
    f_sum(x,y) = x + y
    f_harm(x,y) = 2 * (1.0 / ((1.0 / x + 1.0 / y) / 2.0))
    f_l2(x,y) = 2.0 * sqrt((x^2 + y^2) / 2.0)
    f_max(x, y) = 2.0 * max(x, y)

    # nonlinear first-order functions
    f_p(x) = x .^ 0.5

    methods = [(f_geom, "geometric",  :HOLS),
               (f_sum,  "arithmetic", :HOLS),
               (f_harm, "harmonic",   :HOLS),
               (f_l2,   "L^2",        :HOLS),
               (f_max,  "maximum",    :HOLS),
               (:none,  "standard",   :LS),
               (f_p,    "p0.5",       :NLS)]

    num_rand_trials = 5

    ho_grid = [a for a in Iterators.product(0.3:0.1:0.8, 0.1:0.15:0.9 ) if a[1] + a[2] < 1]
    ls_grid = 0.1:0.1:0.9
    noise = 0.0
    balanced = true
    max_iterations = 40
    num_CV_trials = 5

    gnn_lrs = [1e-2, 1e-3, 1e-4]
    gnn_wds = [0, 1e-4]
    gnn_grid = collect(Iterators.product(gnn_lrs, gnn_wds))

    percentage_of_known_labels = [0.05, 0.10, 0.15, 0.20]
    results = DataFrame()

    A, true_labels  = read_hols_datasets(which)
    A, DG_isqrt, T, DH_isqrt, B, _, _ =
        Tensor_Package.compute_matrix_and_tensor_binary(Float64.(A), noise=noise)

    # Save training / validation splits for external codes
    all_train_inds = Dict{Float64, Vector{Vector{Int64}}}()
    all_val_inds = Dict{Float64, Vector{Vector{Int64}}}()
    for p in percentage_of_known_labels
        all_train_inds[p] = Vector{Vector{Int64}}(undef, num_rand_trials)
        all_val_inds[p] = Vector{Vector{Int64}}(undef, num_rand_trials)
    end

    for trial in 1:num_rand_trials
        for percentage in percentage_of_known_labels
            df, training_inds = CV_binary(A, DG_isqrt, T, DH_isqrt, B, true_labels,
                                          methods,
                                          num_CV_trials,
                                          percentage,
                                          balanced=balanced,
                                          ε=1e-2,
                                          noise=noise,
                                          ho_search_params=ho_grid,
                                          ls_search_params=ls_grid,
                                          max_iterations=max_iterations)

            println(df)
            println("GraphSAGE...")
            for agg_type in [:SAGE_Max, :SAGE_GCN]
                @time acc, rec, prec, train, val =
                    graph_sage_eval(A, true_labels, training_inds, gnn_grid, :onehot, agg_type)
                # Copy data frame results
                df2 = DataFrame(df[end,:])
                df2[1, :method_name] = String(agg_type)
                df2[1, :acc] = acc
                df2[1, :rec] = rec
                df2[1, :prec] = prec
                append!(df, df2)

                if agg_type == :SAGE_GCN
                    all_train_inds[percentage][trial] = train
                    all_val_inds[percentage][trial] = val
                end
            end

            df[!, :trial] .= trial

            results = [results;df]
            println(df)
        end
    end

    # Write out results
    CSV.write("final_results_cv_$which.csv", results)

    # Write out reproducibility info
    bson("train_val_inds_$which.bson",
         Dict("all_train_inds" => all_train_inds,
              "all_val_inds"   => all_val_inds,
              "A"              => A,
              "y"              => true_labels,
              "features"       => "one_hot",
              "learning_rates" => gnn_lrs,
              "weight_decays"  => gnn_wds))
end
