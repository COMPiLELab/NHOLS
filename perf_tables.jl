using CSV
using DataFrames
using StatsBase
using Statistics
using Printf

function make_table(ds1, ds2, add_htv=true)
    df1 = CSV.read("./www_results/final_results_cv_$ds1.csv") |> DataFrame
    df2 = CSV.read("./www_results/final_results_cv_$ds2.csv") |> DataFrame
    
    if add_htv
        df1_HTV = CSV.read("./www_results/results_cv_HTV_$ds1.csv") |> DataFrame
        df1[!, :lambda] .= 0.0
        if "maxiter" in names(df1_HTV)
            df1[!, :maxiter] .= 40
        end
        df1_HTV[!, :α] .= 0.0
        df1_HTV[!, :β] .= 0.0
        append!(df1, df1_HTV)
    end
    
    if add_htv
        df2_HTV = CSV.read("./www_results/results_cv_HTV_$ds2.csv") |> DataFrame
        df2[!, :lambda] .= 0.0
        if "maxiter" in names(df2_HTV)
            df2[!, :maxiter] .= 40
        end
        df2_HTV[!, :α] .= 0.0
        df2_HTV[!, :β] .= 0.0
        append!(df2, df2_HTV)
    end
    
    method_map = Dict("maximum"  => "NHOLS, max",
                      "geometric"  => "NHOLS, geom",
                      "harmonic"   => "NHOLS, harm",
                      "arithmetic" => "NHOLS, arith",
                      "L^2"        => "NHOLS, \$L^2\$",
                      "standard"   => "Standard LS",
                      "HTV"        => "HTV",
                      "p0.5"   => "NFOLS (p = 0.5)",
                      "p0.7"   => "NFOLS (p = 0.7)",
                      "SAGE_Max"   => "GraphSAGE",
                      "SAGE_GCN"   => "GCN")

    ds_map = Dict("mnist"   => "MNIST",
                  "f-mnist" => "Fashion-MNIST",
                  "pendigits" => "pendigits",
                  "optdigits" => "optdigits",
                  "Caltech36" => "Caltech36",
                  "Rice31" => "Rice31")

    methods = ["arithmetic",
               "harmonic",
               "L^2",
               "geometric",
               "maximum",
               "standard",
               "p0.5"]

    if add_htv; push!(methods, "HTV"); end
    push!(methods, "SAGE_GCN", "SAGE_Max")

    frac_labeled1 = sort(unique(df1[!, :percentage_of_known_labels]))[1:4]
    frac_labeled2 = sort(unique(df2[!, :percentage_of_known_labels]))[1:4]
    num1 = df1[1, :size]
    num2 = df2[1, :size]

    perf_lines = []    
    for (i, method) in enumerate(methods)
        df1_method = df1[df1[!, :method_name] .== method, :]
        df2_method = df2[df2[!, :method_name] .== method, :]

        name = method_map[method]
        line = ["$name &"]

        for frac in frac_labeled1
            accs = df1_method[df1_method[!, :percentage_of_known_labels] .== frac, :acc]
            mean_acc = round(mean(accs), digits=1)
            min_acc = round(minimum(accs), digits=1)
            max_acc = round(maximum(accs), digits=1)
            median_acc = round(median(accs),  digits=1)
            std_acc = round(std(accs), digits=1)
            push!(line, "$mean_acc \\smallpm{$(std_acc)}")
        end

        for frac in frac_labeled2
            accs = df2_method[df2_method[!, :percentage_of_known_labels] .== frac, :acc]
            mean_acc = round(mean(accs), digits=1)
            min_acc = round(minimum(accs), digits=1)
            max_acc = round(maximum(accs), digits=1)
            median_acc = round(median(accs),  digits=1)
            std_acc = round(std(accs), digits=1)
            push!(line, "$mean_acc \\smallpm{$(std_acc)}")            
        end
        
        push!(perf_lines, join(line, " & "))
    end

    println(raw"""\begin{tabular}{l  r  cccc   cccc}""")
    println(raw"""\toprule""")
    println(raw"""& & \multicolumn{4}{c}{""", "$(ds_map[ds1]) (n = $num1)",
            raw"""} & \multicolumn{4}{c}{""", "$(ds_map[ds2]) (n = $num2)",
            raw"""} \\\\""")
    println(raw"""\cmidrule(lr){3-6} \cmidrule(lr){7-10}""")
    prcnt_labeled(frac_labeled) = [@sprintf("%s\\%%", "$(round(f * 100, digits=5))") for f in frac_labeled]
    println("method & \\% labeled",
            " & ", join(prcnt_labeled(frac_labeled1), " & "),
            " & ", join(prcnt_labeled(frac_labeled2), " & "),
            raw""" \\\\""")
    println(raw"""\midrule""")    
    for perf_line in perf_lines
        println(perf_line, raw""" \\\\""")
    end
    println(raw"""\bottomrule""")
    println(raw"""\end{tabular}""")    
end

