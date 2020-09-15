using Plots
using ProgressMeter

mutable struct Planet
    position::Vector{Float64}
    velocity::Vector{Float64}
    mass::Float64
    function Planet(p, v, m)
        @assert length(p) == length(v)
        return new(p, v, m)
    end
end

function distance(x, y)
    return sum((x-y).^2)^(3/2)
end

function force_vector(p::Planet, q::Planet)
    return p.mass * q.mass / distance(p.position,q.position) * (q.position - p.position)
end

function force_vector(planet::Planet, planets::Vector{Planet})
    return mapreduce(p -> force_vector(planet, p), +, planets)
end

function simulate_orbits(planets::Vector{Planet}, Δt::Float64, stop_time::Float64; animate=false, every=1)
    N = ceil(stop_time / Δt)
    planet_count = length(planets)
    d = length(first(planets).position)
    for planet in planets
        @assert length(planet.position) == d
    end

    if animate
        anim = Animation()
    end
    plt = plot(legend=false)

    force_vectors = zeros(planet_count, d)
    @showprogress for n in 1:N
        scatter!(map(p -> Tuple(p.position), planets), mc=1:planet_count)#, markerstrokecolor=1:planet_count)
        if animate && (n-1) % every == 0
            frame(anim, plt)
        end
        # calc force vectors
        for (k,planet) in enumerate(planets)
            others = planets[vcat(1:k-1,k+1:planet_count)]
            force_vectors[k,:] = force_vector(planet, others)
        end

        # apply force vectors
        for (k,planet) in enumerate(planets)
            fv = force_vectors[k,:]

            planet.velocity += Δt/planet.mass * fv
            planet.position += Δt * planet.velocity
        end
    end

    if animate
        return anim
    else
        return plt
    end
end


p = Planet([1.,0.,1.], [0, 0.5, 0.25], 1.)
q = Planet([-1.,0.,1.], [0, -0.5, 0.25], 1.)

simulate_orbits([p,q], 0.02, 10.)

gif(anim, "orbits.gif")

p = Planet([1.,0.], [0,0.5], 1.)
q = Planet([-1.,0.], [0,-0.5], 1.)
r = Planet([0.,1], [-0.5,0], 1.)
s = Planet([0,-1.], [0.5,0.], 1.)

anim = simulate_orbits([p,q,r,s], 0.02, 10., animate=true)
gif(anim, "spiral_orbits.gif")

sun = Planet([0.,0.], [0.,0.], 100_000)
p = Planet([100.,0.], [0.,15.], 1.)
q = Planet([50.,0.], [0.,20.], 1.)
r = Planet([-50.,0.], [0.,50.],100.)

anim = simulate_orbits([sun,p,q,r], 0.02, 36., animate=true)
gif(anim, "sun_orbits.gif")
