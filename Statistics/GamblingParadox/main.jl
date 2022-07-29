

function advantage(d::Int=10; a0=missing, b0=missing)
    # numbers chose by player 1
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
    # numbers chose by player 1
    if !ismissing(a0) && !ismissing(b0)
        a = a0
        b = b0
    else
        a = rand() * d
        b = rand() * d
    end

    # x is chosen by player 1 -> always same order
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
