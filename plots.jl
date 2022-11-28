using CSV, DataFrames, PyCall,  Seaborn, LaTeXStrings, Pandas

pygui(:tk)

close("all")






### TIME COMPARE
##################################################################
figure()
df = CSV.read("./submission_material/results_time_compare_ALL.csv")
sizes = unique(df[!, :size])
methods = unique(df[!,:method])

nhols_names = ["geometric", "arithmetic", "harmonic","L^2","maximum"]
times_nhols = zeros(length(sizes))
for (i,size) in enumerate(sizes) 
    for name in nhols_names
        val =  df[ (df[!,:size].==size) .& (df[!,:method].==name), :time][1]
        times_nhols[i] += val
    end
end
times_nhols ./= 5
Seaborn.plot(sizes, times_nhols, label="NHOLS", linewidth=3)

names = ["Standard LS", "HTV","GraphSAGE", "NFOLS, \$p=0.5\$", "GCN"]
for (i,method) in enumerate(methods[6:end])
    times = df[ df[!,:method].==method, :time]
    Seaborn.plot(sizes, times, label=names[i], linewidth=3)
end
legend(loc="lower right", frameon=false)#, labelspacing=0.0)
yscale("log")
xlabel("number of nodes", fontsize=13)
ylabel("Ex time (sec)", fontsize=13)
##################################################################







######### SBM Heatmaps #########################################################
df = CSV.read("./submission_material/final_results_cv_SBM_4.csv")

methods = unique(df[!,:method_name])
methods = methods[1:end-1]

names = ["NHOLS, geom", "NHOLS, arith", "NHOLS, harm", "NHOLS, \$L^2\$", "NHOLS, max", "Standard LS", "NFOLS, \$p=0.5\$", "HTV","GraphSAGE","GCN"]
trials = unique(df[!,:trial])
ratios = unique(df[!,:ratio])
percentages = unique(df[!,:percentage_of_known_labels])


acc = Dict(method=>zeros(length(ratios), length(percentages)) for method in methods)

for method in methods
    for trial in 1:7#trials
        for (j,percentage) in enumerate(percentages)
            for (i,ratio) in enumerate(ratios)
                # @show method, trial, percentage, ratio
                a =  df[ (df[!,:percentage_of_known_labels].==percentage) .& (df[!,:trial].==trial) .& (df[!,:method_name].==method) .& (df[!,:ratio].==ratio) , :acc]
                acc[method][i,j] += a[1]
            end
        end
    end
end

figure()
for (plotnum,method) in enumerate(methods)
    subplot(2,5,plotnum)
    # imshow(acc[method], filternorm=false)
    Seaborn.heatmap(acc[method]./7, annot=true,  xticklabels=Int64.(floor.(percentages.*100)), yticklabels = ratios,
                    vmin=50, vmax=100, cbar = false, linecolor="black", linewidths=.5)
    # yticks([])
    # xticks(ratios)
    if plotnum > 5
        xlabel("% known labels", fontsize=13)
    end
    if plotnum == 1 || plotnum == 6
        ylabel(L"p_{in}/p_{out}", fontsize=13)
    end
    title(names[plotnum], fontsize=14)
end

subplot(2,5,10)
dfgcn = CSV.read("./submission_material/final_results_cv_SBM_only_GCN.csv")
trials = unique(dfgcn[!,:trial])
ratios = unique(dfgcn[!,:ratio])
percentages = unique(dfgcn[!,:percentage_of_known_labels])
acgcn = zeros(length(ratios), length(percentages))
for trial in trials
    for (j,percentage) in enumerate(percentages)
        for (i,ratio) in enumerate(ratios)
            # @show method, trial, percentage, ratio
            a =  dfgcn[ (dfgcn[!,:percentage_of_known_labels].==percentage) .& (dfgcn[!,:trial].==trial)  .& (dfgcn[!,:ratio].==ratio) , :acc]
            acgcn[i,j] += a[1]
        end
    end
end

Seaborn.heatmap(acgcn./length(trials), annot=true, xticklabels=Int64.(floor.(percentages.*100)), yticklabels = ratios,
                vmin=50, vmax=100, cbar = false, linecolor="black", linewidths=.5)
title("GCN", fontsize=14)    
xlabel("% known labels", fontsize=13)
tight_layout()

# colorbar()
