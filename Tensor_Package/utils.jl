export accuracy, precision, recall, train_test_val_split, train_test_split

struct SuperSparse3Tensor
    I::Vector{Int64}
    J::Vector{Int64}
    K::Vector{Int64}
    V::Vector{Float64}
    n::Int64
end


function accuracy(y_predicted, y_actual)
    return ( sum(y_predicted .== y_actual) ./ length(y_actual) )*100
end

function precision(y_predicted, y_actual; method="mean")
    if method=="mean"
        p = 0
        for label in unique(y_predicted)
            p += sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_predicted .== label)
        end
        p = p/length(unique(y_predicted)) *100
    elseif method=="min"
        p = Inf
        for label in unique(y_predicted)
            p = min(p, sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_predicted .== label) ) *100
        end
    else
        @assert false
    end
    return p
end

function recall(y_predicted, y_actual; method="mean")
    if method=="mean"
        p = 0
        for label in unique(y_predicted)
            p += sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_actual .== label)
        end
        p = p/length(unique(y_predicted)) *100
    elseif method=="min"
        p = Inf
        for label in unique(y_predicted)
            p = min(p, sum((y_predicted .== label) .* (y_actual .== label)) / sum(y_actual .== label) ) *100
        end
    else
        @assert false
    end
    return p
end

function train_test_val_split(y, perc_train, perc_val; balanced=false)
    n = length(y)
    num_classes = length(unique(y))
    Y_test = zeros(n, num_classes)
    Y_train = zeros(n, num_classes)
    Y_val = zeros(n, num_classes)
    test_mask = zeros(n)
    train_mask = zeros(n)
    val_mask = zeros(n)
    num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
    num_val_per_class = Tensor_Package.generate_known_labels(perc_val, balanced, y)
    for (label, num_train, num_val) in zip(1:num_classes, num_train_per_class, num_val_per_class)
        class_inds = findall(y .== label)
        shuffle!(class_inds)

        train_indices = class_inds[1:num_train]
        Y_train[train_indices, label] .= 1
        train_mask[train_indices] .= 1

        test_indices = class_inds[num_train+num_val+1:end]
        Y_test[test_indices, label] .= 1
        test_mask[test_indices] .= 1

        val_indices = class_inds[num_train+1:num_train+num_val]
        Y_val[val_indices, label] .= 1
        val_mask[val_indices] .= 1

    end
    return Y_train, Y_test, Y_val, Bool.(train_mask), Bool.(test_mask), Bool.(val_mask)
end

function fill_array(arr, inds1, inds2)
    for (ind1, ind2) in zip(inds1, inds2)
        arr[ind1, ind2] = 1
    end
    return arr
end



function train_test_split(X, y, perc_train; balanced=false)
    n = length(y)
    @show n
    num_classes = length(unique(y))
    num_train_per_class = Tensor_Package.generate_known_labels(perc_train, balanced, y)
    num_train_total = sum(num_train_per_class)
    @show num_train_total
    Y_test = zeros(n - num_train_total, num_classes)
    Y_train = zeros(num_train_total, num_classes)
    full_train_indices = []
    full_test_indices = []
    for (label, num_train) in zip(1:num_classes, num_train_per_class)
        class_inds = findall(y .== label)
        shuffle!(class_inds)

        train_indices = class_inds[1:num_train]
        push!(full_train_indices, train_indices...)
        #Y_train[(label-1)*num_train + 1:label*num_train, label] .= 1

        num_test = length(class_inds) - num_train
        test_indices = class_inds[num_train+1:end]
        push!(full_test_indices, test_indices...)
        #Y_test[(label-1)*num_test + 1:label*num_test, label] .= 1

    end
    Y_train = fill_array(Y_train, 1:size(Y_train)[1], y[full_train_indices])
    Y_test = fill_array(Y_test, 1:size(Y_test)[1], y[full_test_indices])
    @show full_train_indices
    @show y[full_train_indices]
    return Y_train, Y_test, X[full_train_indices, :], X[full_test_indices, :], X[vcat(full_train_indices, full_test_indices), :], y[vcat(full_train_indices, full_test_indices)]
end
