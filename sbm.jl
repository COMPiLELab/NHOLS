function stochastic_block_model(P,block_sizes; seed="default")
    # Input: P = matrix of edge probabilities
    #        block_sizes = vector of sizes of blocks
    #        seed = random seed for reproducibility
    # Output: A = adjacency matrix of the graph (sparse format)
    #         classes = matrix of ground truth classes assignment

    k = length(block_sizes);
    n = sum(block_sizes);
    II = [];
    J = [];

    for i1 in 1:k
        for i2 in 1:block_sizes[i1]
            i = sum(block_sizes[1:i1-1]) + i2; # position in the big matrix

            # diagonal elements
            lriga = block_sizes[i1] - i2;
            if seed != "default"
                Random.seed!(seed);
            end
            e = rand(1,lriga);
            Jnew = findall(e[:] .<= P[i1,i1]);
            enew = length(Jnew);
            if enew > 0
                to_add_to_II = [i for _ in 1:enew];
                II = [II; to_add_to_II];
                J = [J; i.+Jnew];
            end

            for j1 in i1+1:k
                j = sum(block_sizes[1:j1-1]);
                if seed != "default"
                    Random.seed!(seed);
                end
                e = rand(1,block_sizes[j1]);
                Jnew = findall(e[:] .<= P[i1,j1]);
                enew = length(Jnew);
                if enew > 0
                    to_add_to_II = [i for _ in 1:enew];
                    II = [II; to_add_to_II];
                    J = [J; j.+Jnew];
                end
            end

        end
    end

    A = sparse([II;J],[J;II],1,n,n);

    classes = zeros(Float64,n);
    for i in 1:k
        imin = sum(block_sizes[1:i-1]) + 1;
        imax = sum(block_sizes[1:i-1]) + block_sizes[i];
        classes[imin:imax] .= i;
    end

    return A, classes
end
