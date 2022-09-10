

function advantage(d::Int=10; a0=missing, b0=missing)
    # numbers chosen by player 1
    if !ismissing(a0) && !ismissing(b0)
        a = a0
        b = b0
    else
        a = rand() * d
        b = rand() * d
    end

    # x is chosen by player 2
    x, y = rand() < 0.5 ? (a,b) : (b,a)

    # random number to base prediction on
    z = rand() * d

    if x == z
        # edge case -> decide randomly
        c = rand() < 0.5 ? (x, y) : (y, x)
    elseif x < z
        c = (x, y) # predict chosen number is the smaller one
    else # x > z
        c = (y, x) # predict chosen numer is the bigger one
    end

    return c[1] < c[2] # verify if correct prediction
end


function no_advantage(d::Int=10; a0=missing, b0=missing)
    # numbers chosen by player 1
    if !ismissing(a0) && !ismissing(b0)
        a = a0
        b = b0
    else
        a = rand() * d
        b = rand() * d
    end

    # x is chosen by player 1
    if d - min(a,b) < max(a,b)
        x, y = (min(a,b), max(a,b)) # less probable that z is bigger than min
    else
        x, y = (max(a,b), min(a,b)) # less probable that z is smaller than max
    end

    # random number to base prediction on
    z = rand() * d

    if x == z
        # edge case -> decide randomly
        c = rand() < 0.5 ? (x, y) : (y, x)
    elseif x < z
        c = (x, y) # predict chosen number is the smaller one
    else # x > z
        c = (y, x) # predict chosen numer is the bigger one
    end

    return c[1] < c[2] # verify if correct prediction
end

N = 10^8

sum(advantage(a0=8,b0=9) for i in 1:N) / N
sum(no_advantage(a0=8,b0=9) for i in 1:N) / N

sum(advantage(a0=1,b0=2) for i in 1:N) / N
sum(no_advantage(a0=1,b0=2) for i in 1:N) / N

sum(advantage(a0=4,b0=6) for i in 1:N) / N
sum(no_advantage(a0=4,b0=6) for i in 1:N) / N

sum(advantage(a0=5,b0=7) for i in 1:N) / N
sum(no_advantage(a0=5,b0=7) for i in 1:N) / N

using Random
begin
    Random.seed!(0)
    N = 1_000_000
    d = 10
    steps = 0:0.1:1.0
    scores = zeros(Int, length(steps))
    for (i, p) in enumerate(steps), _ in 1:N
        a = rand()*d
        b = rand()*d
        if rand() < p
            a0 = min(a,b)
            b0 = max(a,b)
        else
            a0 = max(a,b)
            b0 = min(a,b)
        end

        scores[i] += advantage(d, a0=a0, b0=b0)
    end
    return collect(zip(steps, scores./N))
end
