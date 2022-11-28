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
include("sbm.jl")


function main()

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

    number_of_CV_trials = 5
    num_rand_trials = 7

    ho_grid = [a for a in Iterators.product(0.3:0.1:0.8, 0.1:0.15:0.9 ) if a[1] + a[2] < 1]
    ls_grid = 0.1:0.1:0.9
    noise = 0.0
    balanced = true
    max_iterations = 40

    gnn_lrs = [1e-2, 1e-3, 1e-4]
    gnn_wds = [0, 1e-4]
    gnn_grid = collect(Iterators.product(gnn_lrs, gnn_wds))

    percentage_of_known_labels = [.06, .09, .12, .15, .18, .21]
    pin_pout_ratio_range = [2, 2.5, 3, 3.5, 4]
    block_sizes = [100,200,400]
    
    results = DataFrame()


    for ratio in  pin_pout_ratio_range
        pin = .1
        pout = pin/ratio
        k = length(block_sizes)
        P = pout .* ones(k,k)
        P = Diagonal((pin - pout) .* ones(k)) + P

        for trial in 1:num_rand_trials
            A, true_labels = stochastic_block_model(P,block_sizes)
            A, DG_isqrt, T, DH_isqrt, B, _, _ = Tensor_Package.compute_matrix_and_tensor_binary(Float64.(A), noise=0)

            for percentage in percentage_of_known_labels
                (df, training_inds) = CV_binary(A, DG_isqrt, T, DH_isqrt, B, true_labels,
                                              methods,
                                              number_of_CV_trials,
                                              percentage,
                                              balanced=balanced,
                                              ε=1e-3,
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
                df[!, :ratio] .= ratio

                results = [results;df]
                println(df)

            end
            
        end          

    end
    CSV.write("final_results_cv_SBM_only_LS.csv", results)

end
