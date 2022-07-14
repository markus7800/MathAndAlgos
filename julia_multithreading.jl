
begin
    N = 100
    @time for i in 1:N
        sleep(0.1)
    end
end

begin
    N = 100
    t = zeros(Int, N)
    @time Threads.@threads for i in 1:N
        sleep(0.1)
        t[i] = Threads.threadid()
    end
    for n in 1:Threads.nthreads()
        print(n, ": ")
        for i in 1:N
            if t[i] == n
                print("*")
            else
                print(" ")
            end
        end
        println()
    end
end
