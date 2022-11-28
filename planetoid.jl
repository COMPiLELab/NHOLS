include("Tensor_Package/Tensor_Package.jl")


using   .Tensor_Package.Tensor_Package,
        Random,
        Base.Threads,
        SparseArrays,
        MLDataUtils,
        PyCall,
        LightGraphs,
        DataStructures,
        MLDatasets,
        CSV,
        BSON,
        DataFrames,
        LinearAlgebra

path_to_planetoid = string(pwd(), "/planetoid-master")


pushfirst!(PyVector(pyimport("sys")."path"), path_to_planetoid)


@pyimport scipy.sparse as sp


function load_data(dataset_name, kn)
    @show dataset_name

    #### UCI datasets ###################
    try
        X, y = Tensor_Package.prepare_uci_data(dataset_name)
        adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
        return X, y, adj_matrix
    catch KeyErrror
        println("$dataset_name is not a part of UCI datasets or the specified name is spelled wrong.")
    end
    ##################################

   #### Matlab datasets ###################
   mat_dataset_names = ["3sources","BBC4view_685","BBCSport2view_544","cora","UCI_mfeat", "citeseer", "WikipediaArticles"]
   #there are not features here, so it return adj_matrix twice
   if dataset_name in mat_dataset_names
       data = MAT.matread("./data/matlab_multilayer_data/"*dataset_name*"/knn_10.mat")
       y = data["labels"][:]
       adj_matrix= data["W_cell"][1]
       return adj_matrix, y, adj_matrix
   else
      println("$dataset_name is not a part of our matlab multilayer data.")
   end
   ##################################

   #### Pendigits ###################
   if dataset_name == "pendigits"
        train = Array(CSV.read("./data/pendigits.csv"))
        X = train[:,1:end-1]
        adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
        y = train[:,end] .+ 1
        return X, y, adj_matrix
   ##################################

   #### Optdigits ###################
   elseif dataset_name == "optdigits"
       train = Array(CSV.read("./data/optdigits.csv"))
       X = train[:,1:end-1]
       adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
       y = train[:,end] .+ 1
       return X, y, adj_matrix
   ##################################


   #### F-MNIST #####################
   elseif dataset_name == "f-mnist"
       train_x, train_y = FashionMNIST.traindata()
       rows, cols, num = size(train_x)
       X = reshape(train_x, (rows*cols, num))'
       X = convert(Array{Float64,2}, X)
       adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
       y = train_y
       return X, y, adj_matrix

   #### MNIST #######################
   elseif dataset_name == "mnist"
       train_x, train_y = MNIST.traindata()
       rows, cols, num = size(train_x)
       X = reshape(train_x, (rows*cols, num))'
       X = convert(Array{Float64,2}, X)
       adj_matrix = Tensor_Package.distance_matrix(X, kn, mode="connectivity")
       y = train_y
       return X, y, adj_matrix
   ##################################
   else
       println("$dataset_name is not one of the digits datasets.")
   end

   if dataset_name in readdir("data/custom/")
       files = readdir("./data/custom/$dataset_name")
       @show files
       if length(files) == 1
           data_file = files[1]
           if endswith(data_file, ".csv")
               data = Array(CSV.read("./data/custom/$dataset_name/$data_file"))
               X = data[:,1:end-1]
               y = data[:,end] .+ 1
               adj_matrix = distance_matrix(X, kn, mode="connectivity")
               return X, y, adj_matrix
           elseif endswith(data_file, ".mat")
               data = MAT.matread("./data/custom/$dataset_name/$data_file")
               y = data["labels"][:]
               adj_matrix= data["W_cell"][1]
               return adj_matrix, y, adj_matrix
           elseif endswith(data_file, r".xls[x]?")
               data = convert(Array, DataFrame(load("./data/custom/$dataset_name/$data_file", split(data_file, '.')[1])))
               X = data[:,1:end-1]
               y = data[:,end] .+ 1
               adj_matrix = distance_matrix(X, kn, mode="connectivity")
               return X, y, adj_matrix
           end
       elseif length(files) == 2
           features_file = filter(x -> startswith(x, 'X'), files)[1]
           labels_file = filter(x -> startswith(x, 'y'), files)[1]
           if endswith(features_file, ".npy")
               X = npzread("./data/custom/$dataset_name/$features_file")
               y = npzread("./data/custom/$dataset_name/$labels_file")
               adj_matrix = distance_matrix(X, kn, mode="connectivity")
               return X, y, adj_matrix
           end
       end
   else
       println("$dataset_name is not one of the datasets you provided.")
   end

   return nothing, nothing
end


function make_graph(K)
    graph = SortedDict()
    for a in collect(edges(Graph(K)))
            python_src_edge = src(a) - 1
            python_dst_edge = dst(a) - 1
            if python_src_edge in keys(graph)
                    push!(graph[python_src_edge], python_dst_edge)
            else
                    graph[python_src_edge] = [python_dst_edge]
            end

            if python_dst_edge in keys(graph)
                    push!(graph[python_dst_edge], python_src_edge)
            else
                    graph[python_dst_edge] = [python_src_edge]
            end
    end
    return graph
end


function train_test_val(y, train_indices, test_indices)
    n = length(y)
    num_classes = length(unique(y))
    Y_test = zeros(n, num_classes)
    Y_train = zeros(n, num_classes)
    test_mask = zeros(n)
    train_mask = zeros(n)
    val_mask = zeros(n)
    Y_train = fill_array(Y_train, train_indices, y[train_indices])
    Y_test = fill_array(Y_test, test_indices, y[test_indices])
    train_mask[train_indices] .= 1
    val_mask[val_indices] .= 1
    test_mask[test_indices] .= 1
    return Y_train, Y_test, Y_val, Bool.(train_mask), Bool.(test_mask), Bool.(val_mask)
end



# model_name can be 'gcn', 'gcn_cheby', 'dense'
function PLANETOID(dataset_name, kn, perc_train; balanced=true)

        args = Dict("learning_rate" => .1,
                     "embedding_size" => 50,
                     "window_size" => 3,
                     "path_size" => 10,
                     "batch_size" => 200,
                     "g_batch_size" => 200,
                     "g_sample_size" => 100,
                     "neg_samp" => 0,
                     "g_learning_rate" => 1e-2,
                     "model_file" => "trans.model",
                     "use_feature" => true,
                     "update_emb" => true,
                     "layer_loss" => true)

        X, y, _ = load_data(dataset_name, kn)


        Y_train, Y_test, X_train, X_test, X, y = Tensor_Package.train_test_split(X, y, perc_train; balanced=balanced)

        K = Tensor_Package.distance_matrix(X, kn, mode="connectivity")

        graph = make_graph(K)

        X_test_python = sp.csr_matrix(X_test)
        X_train_python = sp.csr_matrix(X_train)


        train_JL = pyimport("test_trans_JL")

        test_acc, pred = pyimport("importlib").reload(train_JL)[:train_transductive](dataset_name, args, X_train_python, Y_train, X_test_python, Y_test, graph; max_iter=1000, init_iter_label=500, init_iter_graph=20, tolerance=1e-4)


        return pred, test_acc
end


#py"""exec(open("test_trans.py").read())"""

#pred, test_acc = PLANETOID("optdigits", 15, .01; balanced=true)


function fill_array(arr, inds1, inds2)
    for (ind1, ind2) in zip(inds1, inds2)
        arr[ind1, ind2] = 1
    end
    return arr
end

function train_test_split_v2(X, y, train_batch, test_batch)
    n = length(y)
    num_classes = length(unique(y))
    Y_test = zeros(length(test_batch), num_classes)
    Y_train = zeros(length(train_batch), num_classes)
    Y_test = fill_array(Y_test, 1:length(test_batch), y[test_batch])
    Y_train = fill_array(Y_train, 1:length(train_batch), y[train_batch])
    return Y_train, Y_test, X[train_batch, :], X[test_batch, :], X[vcat(train_batch, test_batch), :], y[vcat(train_batch, test_batch)], vcat(train_batch, test_batch)
end

function planetoid_v2(kn)

    args = Dict("learning_rate" => .1,
                 "embedding_size" => 50,
                 "window_size" => 3,
                 "path_size" => 10,
                 "batch_size" => 200,
                 "g_batch_size" => 200,
                 "g_sample_size" => 100,
                 "neg_samp" => 0,
                 "g_learning_rate" => 1e-2,
                 "model_file" => "trans.model",
                 "use_feature" => true,
                 "update_emb" => true,
                 "layer_loss" => true)

    df = DataFrame(dataset_name = String[], size = Int64[], percentage = Float64[],
                    trial_num = Int64[], learning_rate = Float64[],
                    accuracy = Float64[], recall = Float64[], precision = Float64[])
    for  dataset_name in ["Rice31"]
        X = nothing
        if dataset_name ∉ ["Rice31", "Caltech36"]
            X, y, _ = load_data(dataset_name, kn)
        end
        data = BSON.load("submission_material/NHOLS-data/train_val_inds_$dataset_name.bson")
        n = size(data["A"])[1]
        if isa(data["features"], String)
            X = Matrix(1.0I, n, n)
        end
        for (train_indices, val_indices) in zip(data["all_train_inds"], data["all_val_inds"])
            _, train_batches = train_indices
            percentage, val_batches = val_indices
            accuracies = []
            recalls = []
            precisions = []
            for (trial_num, (train_batch, val_batch)) in enumerate(zip(train_batches, val_batches))
                test_batch = collect(setdiff(setdiff(Set(1:n), Set(train_batch)), Set(val_batch)))
                new_train_batch = collect(union(Set(train_batch), Set(val_batch)))
                Y_train, Y_test, X_train, X_test, X_new, y_new, indices = train_test_split_v2(X, data["y"], new_train_batch, test_batch)
                #K = Tensor_Package.distance_matrix(X_new, kn, mode="connectivity")
                K = data["A"][indices, indices]
                graph = make_graph(K)
                X_test_python = sp.csr_matrix(X_test)
                X_train_python = sp.csr_matrix(X_train)


                train_JL = pyimport("test_trans_JL")

                pred, test_acc = pyimport("importlib").reload(train_JL)[:train_transductive](dataset_name, args,
                 X_train_python, Y_train, X_test_python, Y_test,
                  graph; max_iter=1000, init_iter_label=4500, init_iter_graph=80, tolerance=1e-4)
                final_pred = vec(map(x->x[2], argmax(pred,dims=2)))
                @show Tensor_Package.accuracy(final_pred, data["y"][test_batch])
                push!(df, [dataset_name, n, percentage, trial_num, .1,
                 Tensor_Package.accuracy(final_pred, data["y"][test_batch]),
                  Tensor_Package.precision(final_pred, data["y"][test_batch]),
                   Tensor_Package.recall(final_pred, data["y"][test_batch])])
                CSV.write("outputfile.csv",df)
            end
        end
    end
    return nothing
end


planetoid_v2(7)
