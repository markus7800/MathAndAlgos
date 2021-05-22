
function edit_distance(s1::String, s2::String)
    n1 = length(s1)+1
    n2 = length(s2)+1
    D = Array{Int,2}(undef, n1, n2)
    D[:,1] = 0:(n1-1)
    D[1,:] = 0:(n2-1)

    for i in 2:n1, j in 2:n2
        D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1] + (s1[i-1] == s2[j-1] ? 0 : 2))
    end

    return D[n1,n2]
end
