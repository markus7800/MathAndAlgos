mutable struct Graph
    nodes::Set{Int}
    edges::Set{Tuple{Int,Int}}
end

function Base.show(io::IO, g::Graph)
    println(io, "GRAPH:")
    println(io, "Nodes:")
    for node in g.nodes
        print(io, node, " ")
    end
    println(io)
    println(io, "Edges:")
    for edge in g.edges
        println(io, edge)
    end
end


mutable struct Bag
    graph_nodes::Set{Int}
    parent::Union{Nothing, Bag}
    label::String
end


function Base.show(io::IO, b::Bag)
    print(io, b.label, ": ", b.graph_nodes)
    if !isnothing(b.parent)
        print(io, ", parent: ", b.parent.label)
    end
end

function order_edge_nodes(g::Graph)
    new_edges = Set{Tuple{Int,Int}}()
    for edge in g.edges
        if edge[1] < edge[2]
            push!(new_edges, edge)
        else
            push!(new_edges, (edge[2], edge[1]))
        end
    end
    g.edges = new_edges
end

function are_edge_nodes_ordered(g::Graph)
    for edge in g.edges
        if edge[2] <= edge[1]
            return false
        end
    end
    return true
end



function eliminate(node::Int, g::Graph, bags::Vector{Bag})
    @assert are_edge_nodes_ordered(g)

    neighbours = Set{Int}()
    node_edges = Set{Tuple{Int,Int}}()
    for edge in g.edges
        if edge[1] == node
            push!(neighbours, edge[2])
            push!(node_edges, edge)
        end
        if edge[2] == node
            push!(neighbours, edge[1])
            push!(node_edges, edge)
        end
    end
    println("Eliminate node $node")
    println("\tNeighbours: ", neighbours)
    println("\tEdges: ", node_edges)



    g_new = deepcopy(g)
    delete!(g_new.nodes, node)
    for node_edge in node_edges
        delete!(g_new.edges, node_edge)
        println("\t\tRemove edge: ", node_edge)
    end

    for n1 in neighbours
        for n2 in neighbours
            if n1 < n2
                new_edge = (n1, n2)
                if !(new_edge in g_new.edges)
                    println("\t\tAdd edge: ", new_edge)
                    push!(g_new.edges, new_edge)
                end
            end
        end
    end

    bag_nodes = copy(neighbours)
    push!(bag_nodes, node)

    bag = Bag(bag_nodes, nothing, "Bag $node")
    println("\tCreate bag: $bag")

    for other_bag in bags
        if isnothing(other_bag.parent)
            if node in other_bag.graph_nodes
                other_bag.parent = bag
                println("\t\tSet as parent to ", other_bag.label)
            end
        end
    end

    push!(bags, bag)

    @assert are_edge_nodes_ordered(g_new)

    println()

    return g_new, bag
end

g = Graph(
    Set([1,2,3,4,5,6,7,8,9,10]),
    Set([(1,2), (2,3), (1,3), (3,4), (1,4), (4,5), (4,7),
    (4,8), (5,6), (5,9), (6,9), (6,8), (7,8), (5,9), (9,10), (7,10), (7,9)
    ])
)
are_edge_nodes_ordered(g)

bags = Bag[]
g_new, bag = eliminate(10, g, bags)
g_new, bag = eliminate(9, g_new, bags)
g_new, bag = eliminate(8, g_new, bags)
g_new, bag = eliminate(7, g_new, bags)
g_new, bag = eliminate(2, g_new, bags)
g_new, bag = eliminate(3, g_new, bags)
g_new, bag = eliminate(6, g_new, bags)
g_new, bag = eliminate(1, g_new, bags)
g_new, bag = eliminate(5, g_new, bags)
g_new, bag = eliminate(4, g_new, bags)
