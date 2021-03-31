using Printf

function simplex(α, β, c, I)
   m, n = size(α)
   @assert length(b) == m
   @assert length(c) == n
   indexes = collect(1:n)

   notI = setdiff(indexes, I) # A[:,I] lin. independent, x[notI] = 0

   Δ = zeros(length(notI))
   for j in 1:length(notI)
      ν = notI[j]
      Δ[j] = c[ν] - (c[I])'α[:,ν]
   end
   ν = notI[argmin(Δ)]

   if all(Δ .>= 0)
      println("Optimum found.")
      return α, β, I
   end


   plus = α[:,ν] .> 0
   I_plus = I[plus]
   if isempty(I_plus)
      println("LP is unbounded.")
      return α, β, I
   end

   E = β[plus] ./ α[plus, ν]
   κ = I_plus[argmin(E)]
   swap_index = findfirst(I .== κ)

   new_I = copy(I)
   new_I[swap_index] = ν

   println("Swap nu=$ν for kappa=$κ at $swap_index")

   # swap κ with ν
   new_α = copy(α)
   new_β = copy(β)

   for j in notI
      new_α[swap_index,j] = α[swap_index,j]/α[swap_index,ν]
   end
   new_α[swap_index,κ] = 1/α[swap_index,ν]
   new_β[swap_index] = β[swap_index] / α[swap_index,ν]

   for i in 1:m
      i == swap_index && continue
      for j in notI
         new_α[i,j] = α[i,j] - α[i,ν] * α[swap_index,j] / α[swap_index,ν]
      end
      new_α[i,κ] = -α[i,ν]/α[swap_index,ν]

      new_β[i] = β[i] - α[i,ν] / α[swap_index,ν] * β[swap_index]
   end

   for (row, i) in enumerate(new_I)
      print("$(new_α[row,i]) x$i")
      for j in 1:n
         i == j && continue
         v = new_α[row,j]
         v ≈ 0 && continue

         print(" + $v x$j")
      end
      println(" = $(new_β[row])")
   end

   return new_α, new_β, new_I
end

A = [
   1  1  0 -1 -1 -2;
  -2 -2  1  0  0 -1;
   3  2 -1  2  1  4
]

b = [-3, 2, 3]

c = [-3, 0, 0, -1, 0, -3]

m, n = size(A)

I = [2,3,5]
α = A[:,I] \ A
β = A[:,I] \ b

α, β, I = simplex(α, β, c, I)

x = zeros(n)
x[I] = β

x

c'x
