using BenchmarkTools
using Random

n = Threads.nthreads()

Random.seed!(0)
A = rand(10^8)
s1 = sum(A)

@btime sum(A) # 46.889 ms (1 allocation: 16 bytes)

@btime begin
    subSums = zeros(n)
    L = length(A) ÷ n
    Threads.@threads for i in 1:n
        range = (L*(i-1)+1):L*i
        subSums[i] = sum(A[range])
    end
    sum(subSums)
end # 139.844 ms (42 allocations: 762.94 MiB)

@btime begin
    subSums = zeros(n)
    L = length(A) ÷ n
    Threads.@threads for i in 1:n
        subSums[i] = sum(@view A[(L*(i-1)+1):L*i])
    end
    sum(subSums)
end # 40.948 ms (38 allocations: 2.69 KiB)




Random.seed!(0)
A = rand(10^8)
L = length(A) ÷ n
As = [A[(L*(i-1)+1):L*i] for i in 1:n]

@btime sum(As[1]) # 11.486 ms (1 allocation: 16 bytes)

@btime sum(As[1])+sum(As[2])+sum(As[3])+sum(As[4]) # 47.388 ms (5 allocations: 80 bytes)

@btime begin
    subSums = zeros(n)
    Threads.@threads for i in 1:n
        subSums[i] = sum(As[i])
    end
    sum(subSums)
end # 42.181 ms (28 allocations: 2.33 KiB)

using Distributed

Distributed.nprocs()
Distributed.addprocs(3)

@btime begin
    r1 = @spawnat 1 sum(As[1])
    r2 = @spawnat 2 sum(As[2])
    r3 = @spawnat 3 sum(As[3])
    r4 = @spawnat 4 sum(As[4])
    fetch(r1) + fetch(r2) + fetch(r3) + fetch(r4)
end # 58.060 ms (334 allocations: 12.86 KiB)

@btime begin
    r1 = @spawnat 1 sum(rand(L))
    r2 = @spawnat 2 sum(rand(L))
    r3 = @spawnat 3 sum(rand(L))
    r4 = @spawnat 4 sum(rand(L))
    fetch(r1) + fetch(r2) + fetch(r3) + fetch(r4)
end # 159.437 ms (342 allocations: 190.75 MiB)

@btime sum(rand(L*4)) # 245.754 ms (4 allocations: 762.94 MiB)


@btime begin
    s = 0.
    for i in 1:length(A)
        s += A[i]
    end
    s
end # 16.318 s (499998980 allocations: 8.94 GiB)


@btime begin
    s = 0.
    for a in A
        s += a
    end
    s
end # 12.686 s (399999490 allocations: 7.45 GiB)

@btime begin
    subSums = zeros(n)
    for i in 1:length(A)
        s += A[i]
    end
    s
end
