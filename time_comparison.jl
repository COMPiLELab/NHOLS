include("Tensor_Package/Tensor_Package.jl")


using .Tensor_Package.Tensor_Package
using LinearAlgebra
using SparseArrays
using MATLAB
using DataFrames
using StatsBase
using Random
using CSV

include("graph_sage.jl")
include("sbm.jl")


function tt(A::SparseArrays.SparseMatrixCSC{Float64,Int64}; weight_function = (x,y,z)-> 1)
    # INPUT A = adjacency matrix (knn matrix) weighted with distances (ie. sparsified distance matrix)

    n = size(A, 1)  # number of nodes
    d = vec(sum(A, dims=2))  # degree vector
    deg_order = zeros(Int64, n)
    deg_order[sortperm(d)] = 1:n

    Is = []
    Js = []
    Ks = []
    Vs = []

    # Incidence matrix and degree vector for Matlab (Matthias code)
    INC_I = []
    INC_J = []
    w = []
    c = 1 # counter for INC_J

    for i = 1:n
        N_i = findnz(A[:, i])[1]  # neighbors of node i
        # only look over pairs of neighbors with higher degree order
        N_i_keep = [j for j in N_i if deg_order[j] > deg_order[i]]
        for jj = 1:length(N_i_keep)
            for kk = (jj + 1):length(N_i_keep)
                j = N_i_keep[jj]
                k = N_i_keep[kk]
                # check for triangle
                if A[j, k] > 0
                    # triangle (i, j, k)
                    #push!(triangles, (i, j, k))
                    v = weight_function(A[i, j], A[j, k], A[k, i])
                    push!(Is, i,i,j,j,k,k)
                    push!(Js, j,k,i,k,i,j)
                    push!(Ks, k,j,k,i,j,i)
                    push!(Vs, v,v,v,v,v,v)

                    push!(INC_I, i,j,k)
                    push!(INC_J, c,c,c)
                    push!(w, v)
                    c += 1

                end
            end
        end
    end
    T = Tensor_Package.SuperSparse3Tensor(Is,Js,Ks,Vs,n)
    INC = sparse(INC_I,INC_J,1, n, length(w))
    return T, INC, w
end


function generate_Y(percentage_of_known_labels, ground_truth_classes)
    number_of_classes = length(unique(ground_truth_classes))
    num_per_class = [sum(ground_truth_classes .== i) for i in 1:number_of_classes]

    known_labels_per_each_class = Int64.(ceil.(percentage_of_known_labels.*num_per_class'))

    Y = zeros(length(ground_truth_classes))

    for (i,class) in enumerate(unique(ground_truth_classes))
        ind_class = findall(ground_truth_classes .== class)
        Y[ind_class[1:known_labels_per_each_class[i]]] .= class
    end

    return Int64.(Y), known_labels_per_each_class
end


function make_tensors_and_mx(W)
    # T,INC,w = time_neighbor_neighbor_tensor(W)
    T,INC,w = tt(Float64.(W))


    DT,DH_isqrt = Tensor_Package.row_normalize(T, W)
    DA,DG_isqrt = Tensor_Package.row_normalize(W)
    B = Tensor_Package.B_matrix(DT)

    return DT, DA, B, INC', w, DH_isqrt, DG_isqrt
end


function make_random_test_graph(blocks, p)
    q = p/3
    k = length(blocks)
    P = q .* ones(k,k)
    P = Diagonal((p - q) .* ones(k)) + P

    A,true_labels = stochastic_block_model(P,blocks)

    return A, Int64.(true_labels)
end


function main()
    Random.seed!(1234)


    f1(x,y) = 2.0 * sqrt(x * y)
    f2(x,y) = x + y
    f3(x,y) = 2.0 * sqrt((x^2 + y^2) / 2.0)
    f4(x,y) = 2 * (1.0 / ((1.0 / x + 1.0 / y) / 2.0))
    f_max(x, y) = 2.0 * max(x, y)

    mixing_functions = [f1, f2, f4, f3, f_max]
    # method_names = ["geometric", "arithmetic",  "harmonic", "L^2", "maximum",  "standard", "NLS", "HTV", "GNN", "GCN"]
    method_names = ["geometric", "arithmetic",  "harmonic", "L^2", "maximum",  "standard", "NLS", "GNN", "GCN"]

    function φ(x, f, B::SparseArrays.SparseMatrixCSC)
        sum = 0.0
        for (i, j, v) in zip(findnz(B)...)
            @inbounds sum += (v * f(x[i], x[j]))^2
        end
        return 0.5 * sqrt(sum)
    end
    # sizes = [30*(2^m) for m in 0:1:5 ]
    # sizes = [100,100,300,1000,3000]
    sizes = [300,1434,2568,3699,4833,5967,7101,8232,9366,10500,12327,13665,15000]

    # append!(sizes, Int64.(round.(LinRange(25,100,3))) )
    times = Dict(method => zeros(length(sizes),1) for method in method_names)
    densities = zeros(length(sizes), 1)
    num_trials = 10

    results = DataFrame()

    for (i,n) in enumerate(sizes)
        for trial in 1:num_trials
            p = 0.05
            println("sbm prob edges = $p")
            A,true_labels = make_random_test_graph([n, n, n], p)
            DT, DA, B, INC, w , DH_isqrt, DG_isqrt= make_tensors_and_mx(A)

            densities[i] += nnz(A)/(6*n)

            Y,_ = generate_Y(.2, true_labels)

            α = rand()
            β = rand()*(1-α)
            while α + β >= 1 - 1e-5
                α = rand()
                β = rand()*(1-α)
            end

            γ = 1-α-β
            tol = 1e-4
            maxit = 150
            verbose = false

            ϵ = 1e-4
            Y1 = zeros(length(true_labels)); Y1[Y.==1].=1
            Yϵ = (1-ϵ) .* Y1 .+ ϵ

            ## NHOLS ##################################
            for (j,f) in enumerate(mixing_functions)
                t0 = time()
                X_HOLS, err_HOLS, it_HOLS, _ =
                    Tensor_Package.projected_second_order_label_spreading(
                        x -> Tensor_Package.Tf(DT, DH_isqrt, f, x),
                        x -> Tensor_Package.Ax(DA, DG_isqrt, x),
                        Yϵ./Tensor_Package.φ_base(DH_isqrt .* Yϵ, f, B), #φ(Yϵ,f,B),
                        α,β,γ,
                        x->Tensor_Package.φ_base(DH_isqrt .* x, f, B),
                        max_iterations=maxit)
                        
                times[method_names[j]][i] += (time() - t0)

                @show "NHOLS", f, trial, n, densities[i]
                # end
            end
            ###########################################



            ## LS #####################################
            t0 = time()
            X_LS, err_LS, it_LS = Tensor_Package.standard_label_spreading(
                                        x -> Tensor_Package.Ax(DA, DG_isqrt, x),
                                        Y1, α+β, γ,
                                        max_iterations=maxit)    

            times["standard"][i] += (time() - t0)
            ###########################################
            @show "LS", trial, n , densities[i]

            
            ## NLS #####################################
            t0 = time()
            X_LS, err_LS, it_LS = Tensor_Package.standard_label_spreading(
                                                            x -> Tensor_Package.Px(DA, DG_isqrt, σ.(x)),
                                                            Y1, α+β, γ, 
                                                            max_iterations = maxit,
                                                            normalize=true)  
            times["NLS"][i] += (time() - t0)
            ###########################################
            @show "NLS", trial, n , densities[i]

            
            ## HTV ####################################
            lambda = γ / 1 - γ
            m_w = mxarray(Float64.(vec(w)))
            m_INC = mxarray(Float64.(INC))
            m_y = mxarray(Float64.(Y))
            t0 = time()
            X_HTV = mat"hypergraph_TV_2($m_INC,$m_w,$lambda,$m_y,$maxit)"
            times["HTV"][i] += time() - t0
            ############################################
            @show "HTV", trial, n , densities[i]


            ## GraphSAGE ###############################
            # GraphSAGE is too expensive, run only two trials
            if trial < 3
                hyperparams = 1e-3, 1e-4
                N = length(true_labels)
                train_plus_val = findall(Y.!=0)
                test_inds = findall(Y.==0)

                all_labels = true_labels

                for (str_name,agg_type) in zip(["GNN","GCN"],[:SAGE_Max, :SAGE_GCN])        
                    t0 = time()
                    sample_size = 25
                    # acc, rec, prec, train, val =
                    #     graph_sage_eval(A, all_labels, train_plus_val, test_inds, hyperparams, :onehot, agg_type)#(A, true_labels, training_inds, gnn_grid, :onehot, agg_type)
                    acc, prec, rec = run_graph_sage(A, all_labels, train_plus_val, test_inds, hyperparams, :onehot, agg_type, sample_size)
                    # run_graph_sage(A, all_labels, train_plus_val, test_inds, hyperparams, :onehot)
                    times[str_name][i] += (time()-t0)*5 # we then divide by 10 to make the average
                    @show str_name, trial, n, densities[i]
                end
            end
            ###########################################

        end

        df = DataFrame()
        df[:,:size] = (n*3).*ones(length(method_names),1)[:]
        println(df)
        df[:,:density] = (densities[i]/num_trials) .* ones(length(method_names),1)[:]
        df[:,:time] = [times[method][i]/num_trials for method in method_names][:]
        df[:,:method] = method_names[:]
        results = [results;df]
        println(results)

        CSV.write("results_time_comparison_2.csv", results)
    end

    CSV.write("results_time_comparison_2.csv", results)
    println("results_time_comparison_2.csv")
    
end
