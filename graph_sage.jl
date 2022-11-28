using GraphSAGE  # add https://github.com/000Justin000/GraphSAGE
using LightGraphs
using StatsBase
using LinearAlgebra
using SparseArrays
using Arpack

using Flux
using Flux: onehotbatch, onehot, onecold, softmax, logitcrossentropy, throttle, params, @epochs
using Flux.Optimise

function run_graph_sage(A, all_labels, train_inds, test_inds, hyperparams, F,
                        agg_type, sample_size::Int64, num_epochs=15)
    G = Graph(A)
    n = size(A, 1)
    num_labels = maximum(all_labels)

    # Create one-hot encoding model with logistic regression on output representation
    dim_in =
        if F == :onehot
            n
        else
            size(F,1)
        end
    dim_out = num_labels
    dim_hid = num_labels
    enc = graph_encoder(dim_in, dim_out, dim_hid, repeat([String(agg_type)], 2);
                        ks=[sample_size, sample_size], σ=relu);
    final_layer = Dense(dim_out, dim_out)
    model =
        if F == :onehot
            M -> final_layer.(enc(G, M, u -> onehot(u, 1:n)))
        else
            M -> final_layer.(enc(G, M, u -> vec(F[:, u])))
        end

    # Loss function (note: no softmax above, so logitcrossentropy)
    function loss(X, Y)
        rep_X = model(X)
        l = 0.0
        for (i, rep_x) in enumerate(rep_X)
            l += logitcrossentropy(rep_x, Y[:, i])
        end
        return l
    end

    # Create mini-batches
    num_train = length(train_inds)
    shuffle!(train_inds)
    mb_size = min(Int(ceil(length(train_inds) * 0.1)), 10)
    cnt = 1
    training_data = []
    train_labels = all_labels[train_inds]
    while cnt < num_train
        end_ind = min(cnt + mb_size - 1, num_train)
        mb_X = train_inds[cnt:end_ind]
        mb_Y = onehotbatch(all_labels[mb_X], 1:num_labels)
        push!(training_data, (mb_X, mb_Y))
        cnt += mb_size
    end

    Y_train = onehotbatch(all_labels[train_inds], 1:num_labels)
    function call_back1()
        loss_all = loss(train_inds, Y_train) / length(train_inds)
        rep_X = model(train_inds)
        Y_pred = [onecold(rep) for rep in rep_X]
        correct = sum(Y_pred .== all_labels[train_inds])
        acc = correct / length(train_inds)
        println("loss: $loss_all accuracy: $acc")
        return acc
    end
    function call_back2()
        rep_X = model(train_inds)
        Y_pred = [onecold(rep) for rep in rep_X]
        correct = sum(Y_pred .== all_labels[train_inds])
        acc = correct / length(train_inds)
        return acc
    end
    
    lr, wd = hyperparams
    θ = params(enc, final_layer)
    opt = ADAMW(lr, (0.9, 0.999), wd)
    # num_epochs = 15
    for _ in 1:num_epochs
        train!(loss, θ, training_data, opt)
        acc = call_back1()
        if acc > 0.95; break; end
    end

    Y_pred = [onecold(model([t])[1]) for t in test_inds]
    Y_real = all_labels[test_inds]
    return [Tensor_Package.accuracy(Y_pred, Y_real),
            Tensor_Package.recall(Y_pred, Y_real),
            Tensor_Package.precision(Y_pred, Y_real)]
end

function graph_sage_eval(A, all_labels, training_inds, grid, F, agg_type,num_epochs=15)
    A = max.(A, A')
    label_splits = CV_splits(training_inds, 1)
    train = ∪([x[1] for x in label_splits[1]]...)
    val = ∪([x[2] for x in label_splits[1]]...)

    sample_size = 25
    
    # Hyperparameter sweep
    num_grid_pts = length(grid)
    sweep_accs = zeros(Float64, num_grid_pts)
    for i = 1:num_grid_pts
        hyperparams = grid[i]
        sweep_accs[i] = run_graph_sage(A, all_labels, train, val, hyperparams,
                                       F, String(agg_type), sample_size,num_epochs=num_epochs)[1]
        println("acc: $(sweep_accs[i])")
    end
    
    best_hyperparams = grid[argmax(sweep_accs)]
    println("best hyperparameters: $(best_hyperparams)")
    train_plus_val = ∪(train, val)
    test_inds = ones(Bool, size(A, 1))
    test_inds[train_plus_val] .= false
    test_inds = findall(test_inds)
    acc, prec, rec =
    
    run_graph_sage(A, all_labels, train_plus_val, test_inds, best_hyperparams,
                       F, agg_type, sample_size)
    
    return acc, prec, rec, train, val
    
end
