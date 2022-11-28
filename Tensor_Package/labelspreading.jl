export generate_known_labels, φ, projected_second_order_label_spreading, standard_label_spreading

struct SuperSparse3Tensor
    I::Vector{Int64}
    J::Vector{Int64}
    K::Vector{Int64}
    V::Vector{Float64}
    n::Int64
end

function projected_second_order_label_spreading(Tfun,Afun,y,α,β,γ,φ;
                                                x_0=y,
                                                max_iterations = 200,
                                                tolerance = 1e-5,
                                                verbose = false)
    ## Input: Tfun = function that performs the product Tf(x)
    ##        Afun = funciton that perfroms the product A*x
    error_sequence = [];
    x_new = [];
    φ_values = []
    for k in 1 : max_iterations
        xx_new =  α .* Tfun(x_0) + β .* Afun(x_0) + γ .* y;
        x_new = xx_new ./ φ(xx_new);
        append!(φ_values, φ(xx_new))
        error_sequence = [error_sequence; norm(x_new - x_0) / norm(x_new)];
        if error_sequence[end] < tolerance
            return x_new, error_sequence, k, φ_values
        end

        if φ(xx_new)==NaN
            error("xx_new  or φ(xx_new) are NaN")
        end

        x_0 = x_new
        if verbose
            @show k
        end
    end
    if verbose
        println("Reached max number of iterations without convergence")
    end
    return x_new, error_sequence, max_iterations, φ_values
end

function standard_label_spreading(Afun,y,α,β; x_0=y, max_iterations=200, tolerance=1e-5,
                                  normalize=false, verbose=false)
    ## Input: Afun = funciton that perfroms the product A*x
    error_sequence = [];
    x_new = [];
    for k in 1:max_iterations
        x_new =  α .* Afun(x_0) + β .* y;
        error_sequence = [error_sequence; norm(x_new - x_0) / norm(x_new)];
        if error_sequence[end] < tolerance
            return x_new, error_sequence, k
        end
        if normalize
            x_new /= norm(x_new, 1)
        end
        x_0 = x_new
        if verbose @show k end
    end
    if verbose println("Reached max number of iterations without convergence") end
    return x_new, error_sequence, max_iterations
end

function generate_known_labels(percentage_of_known_labels, balanced, ground_truth_classes)
    number_of_classes = length(unique(ground_truth_classes))
    num_per_class = [sum(ground_truth_classes .== i) for i in 1:number_of_classes]

    if balanced
        known_labels_per_each_class = Int64.(ceil.(percentage_of_known_labels.*num_per_class'))
    else
        known_labels_per_each_class = percentage_of_known_labels.*num_per_class'
        known_labels_per_each_class = Int64.(ceil.(known_labels_per_each_class .+ (known_labels_per_each_class./2).*randn(size(known_labels_per_each_class)) ))
        known_labels_per_each_class = min.(known_labels_per_each_class, num_per_class')
        known_labels_per_each_class = max.(known_labels_per_each_class,1)
    end

    return known_labels_per_each_class
end

function φ_base(x, f, B::SparseArrays.SparseMatrixCSC)
    sum = 0.0
    for (i, j, v) in zip(findnz(B)...)
        @inbounds sum += v * (f(x[i], x[j]))^2
    end
    return 0.5 * sqrt(sum)
end
