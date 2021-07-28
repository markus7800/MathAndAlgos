
using Statistics
using Printf

function pearson_correlation(ru, rv)

    Iu = findall(.!isnan.(ru))
    Iv = findall(.!isnan.(ru))

    println("Iu: ", Iu)
    println("Iv: ", Iv)

    Iuv = intersect(Iu, Iv)
    println("Iuv: ", Iuv)

    mean_ru = 0 #mean(ru[Iu])
    mean_rv = 0 # mean(rv[Iv])

    println(@sprintf "means: ru %.2f rv %.2f" mean_ru mean_rv)

    sd_ru = sqrt(sum((ru[Iu] .- mean_ru).^2))
    sd_rv = sqrt(sum((rv[Iv] .- mean_rv).^2))

    sd_ruv = sum((ru[Iuv] .- mean_ru).*(rv[Iuv] .- mean_rv))

    println(@sprintf "sds: ru %.2f rv %.2f ruv %.2f" sd_ru sd_rv sd_ruv)

    wuv = sd_ruv / (sd_ru * sd_rv)

    println(@sprintf "wuv %.2f" wuv)

end

pearson_correlation(
    [NaN,   1, NaN, -.5, NaN, NaN,   0],
    [NaN,   2, NaN,   1, NaN, NaN,   1])


function pr(class, prediction)
    TP = sum(class[prediction] .== prediction[prediction])

    FP = sum(class[prediction] .!= prediction[prediction])

    #FN = sum(class[.!prediction] .!= prediction[.!prediction])

    FN = sum(class[class] .!= prediction[class])

    return TP / (TP + FP), TP / (TP + FN)
end

function precision_recall(class)

    for j in 1:length(class)
        TP = sum(class[1:j])
        FP = sum(.!class[1:j])
        FN = sum(class[j+1:end])

        println(@sprintf "%d %.2f %.2f\t" j TP / (TP + FP) TP / (TP + FN))
    end
end

precision_recall(
    [true, false, true, true, false, true, false, false, false, true]
)

recommendations = [94, 6, 41, 69, 4, 3, 76, 21, 40, 8, 24, 39, 88, 23, 15, 37, 49, 71, 56, 51]
ground_truth = [6, 21]

precision_recall([r in ground_truth for r in recommendations])



recommendations = [1,2,3,4,5,6,7,100,101,102,103,104,105,106,107,108,109]
ground_truth = collect(1:17)

precision_recall([r in ground_truth for r in recommendations])



function average_precision(class, P, R)
    println(P[class], " ", sum(class))
    mean(P[class])
end

average_precision(
    [true, false, true, true, false, true, false, false, false, true],
    [1, 0.5, 0.667, 0.75, 0.6, 0.667, 0.571, 0.5, 0.444, 0.5],
    [0.2, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 1.]
)

function NDCG(given_ranking)
    println("length: ", length(given_ranking))
    println("given ranking: ")
    for r in given_ranking
        print(@sprintf "%.2f\t" r)
    end
    println()

    for (j, r) in enumerate(given_ranking)
        print(@sprintf "%.2f\t" r / log2(j+1))
    end
    println()

    for j in 1:length(given_ranking)
        print(@sprintf "%.2f\t" sum(given_ranking[k] / log2(k+1) for k in 1:j))
    end
    println()

    ideal_ranking = sort(given_ranking, rev=true)
    println("ideal ranking: ")
    for r in ideal_ranking
        print(@sprintf "%.2f\t" r)
    end
    println()

    for (j, r) in enumerate(ideal_ranking)
        print(@sprintf "%.2f\t" r / log2(j+1))
    end
    println()

    for j in 1:length(ideal_ranking)
        print(@sprintf "%.2f\t" sum(ideal_ranking[k] / log2(k+1) for k in 1:j))
    end
    println()

    println("NDCG: ")
    for j in 1:length(ideal_ranking)
        print(@sprintf "%.3f\t" sum(given_ranking[k] / log2(k+1) for k in 1:j) / sum(ideal_ranking[k] / log2(k+1) for k in 1:j))
    end
    println()
end

NDCG([5, 0, 4, 5, 0, 4, 0, 0, 0, 3])


function borda_count(ratings)
    for r in ratings
        b = sum(r .> ratings) + 0.5 * (sum(r .== ratings) - 1)
        print(b, "\t\t")
    end
    println()
end

borda_count([10, 4, 3, 6, 10, 9, 6, 8, 10, 8])
borda_count([1, 9, 8, 9, 7, 9, 6, 9, 3, 8])
borda_count([10, 5, 2, 7, 9, 8, 5, 6, 7, 6])

function copeland(ratingss)
    sums = zeros(Int, length(ratingss[1]))
    for j in 1:length(ratingss[1])
        for i in 1:length(ratingss[1])
            worse = sum(ratings[j] >= ratings[i] for ratings in ratingss)
            better = sum(ratings[j] <= ratings[i] for ratings in ratingss)
            if worse < better
                print("+\t")
                sums[i] += 1
            elseif worse > better
                print("-\t")
                sums[i] -= 1
            else
                print("0\t")
            end
        end
        println()
    end
    for i in 1:length(ratingss[1])
        print(sums[i], "\t")
    end
    println()
end

copeland([
    [10, 4, 3, 6, 10, 9, 6, 8, 10, 8],
    [1, 9, 8, 9, 7, 9, 6, 9, 3, 8],
    [10, 5, 2, 7, 9, 8, 5, 6, 7, 6]
    ])



function approval(ratingss, threshold)
    R = hcat(ratingss...)
    display(R)
    display(sum(R .> threshold, dims=2))
end

approval([
    [10, 4, 3, 6, 10, 9, 6, 8, 10, 8],
    [1, 9, 8, 9, 7, 9, 6, 9, 3, 8],
    [10, 5, 2, 7, 9, 8, 5, 6, 7, 6]
    ], 6)


function leastmisery(ratingss)
    R = hcat(ratingss...)
    display(R)
    display(minimum(R, dims=2))
end

leastmisery([
    [10, 4, 3, 6, 10, 9, 6, 8, 10, 8],
    [1, 9, 8, 9, 7, 9, 6, 9, 3, 8],
    [10, 5, 2, 7, 9, 8, 5, 6, 7, 6]
    ])



function mostpleasure(ratingss)
    R = hcat(ratingss...)
    display(R)
    display(maximum(R, dims=2))
end

mostpleasure([
    [10, 4, 3, 6, 10, 9, 6, 8, 10, 8],
    [1, 9, 8, 9, 7, 9, 6, 9, 3, 8],
    [10, 5, 2, 7, 9, 8, 5, 6, 7, 6]
    ])


function AverageWithoutMisery(ratingss, threshold)
    R = hcat(ratingss...)
    display(R)
    s = sum(R, dims=2)
    s[sum(R .< threshold, dims=2) .> 0] .= 0
    display(s)
end

AverageWithoutMisery([
    [10, 4, 3, 6, 10, 9, 6, 8, 10, 8],
    [1, 9, 8, 9, 7, 9, 6, 9, 3, 8],
    [10, 5, 2, 7, 9, 8, 5, 6, 7, 6]
    ], 4)



#==============================================================================#




X = hcat(
    [10, 4, 5, 6, 2, 3, 4, 5, 7],
    [5, 7, 8, 7, 5, 7, 9, 1, 6],
    [8, 8, 7, 6, 2, 6, 10, 2, 9]
    )


prod(X, dims = 2)

@sprintf "%.2f" 7 / (7 + 10)


P = 0.23
R = 0.22

2/(1/P + 1/R)

round(2 * P*R / (P + R), digits=2)

@sprintf "%.2f" 2 * P*R / (P + R)

@sprintf "%.2f" 2/(1/P + 1/R)



@sprintf "%.2f" mean([1, 4, 5, 4, 5, 1, 5, 4, 3, 4, 4, 3])

r11 = mean([3, 5, 5, 5, 5, 5, 3, 3])
r14 = mean([4, 5, 5, 5, 5, 5, 5, 5, 4])
r20 = mean([3, 4, 1, 4, 4, 4, 5, 4, 4, 1, 5])

r1 = mean([2, 5, 4])

rn11 = 5
rn14 = 5
rn20 = 4

w11 = 0.4
w14 = 0.3
w20 = 0.3

r1 + (w11 * (rn11 - r11) + w14 * (rn14 - r14) + w20 * (rn20 - r20)) / (w11 + w14 + w20)
