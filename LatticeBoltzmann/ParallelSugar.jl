using MPI

"Structure holding the home-made process Topology"
struct Topology
    rank::Int64
    nworkers::Int64
    dims::Vector{Int64}
    graph::Matrix{Int64}
end

"Constructor of a topology from a MPI.Comm"
function Topology(comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nworkers = MPI.Comm_size(comm)
    dims = [0,0]
    MPI.Dims_create!(nworkers,2,dims)
    Topology(rank,nworkers,dims,collect(reshape(0:nworkers-1,tuple(dims...))))
end

"Returns the position of a process inside the topology given its rank."
position(rank::Int64, topo::Topology)::Int64 = topo.graph[rank+1]
"Returns the position in the Topology of the process calling it."
position(topo::Topology)::Int64 = topo.graph[topo.rank+1]

"Gives the rank of the process N processes to the left(-shift) or to the right(+shift) of the reference rank."
leftright(rank::Int64, shift::Int64, topo::Topology) = topo.graph[mod(rank+shift*topo.dims[1],*(topo.dims...))+1]
"Gives the rank of the process N processes to the left(-shift) or to the right(+shift) of the process calling it."
leftright(shift::Int64, topo::Topology) = leftright(topo.rank, shift::Int64, topo::Topology)
"Returns the number of processes up of the given rank."
numberup(rank::Int64,topo::Topology) = mod(rank,topo.dims[1])
numberup(topo::Topology) = numberup(topo.rank,topo::Topology)
"Returns the number of processes down of the given rank."
numberdown(rank::Int64,topo::Topology) = topo.dims[1] - numberup(rank,topo) -1
numberdown(topo::Topology) = numberdown(topo.rank,topo)
"Returns the number of processes down of the given rank."
numberleft(rank,topo::Topology) = rank ÷ topo.dims[1]
numberleft(topo::Topology) = topo.rank ÷ topo.dims[1]

down(rank::Int64, shift::Int64,topo::Topology) = topo.graph[rank+shift+1]
up(rank::Int64, shift::Int64,topo::Topology) = topo.graph[rank-shift+1]
"Gives the rank of the process N processes up(-shift) or down(+shift) of the process calling it. Fails when out of bounds"
function updown(rank::Int64, shift::Int64,topo::Topology)
    if shift > 0 
        if numberdown(rank,topo) < shift
            throw(DomainError(topo, "There is no process $shift-down from $rank"))
        else
            return down(rank,shift,topo)
        end
    elseif shift < 0
        shift = abs(shift)
        if numberup(rank,topo) < shift
            throw(DomainError(topo, "There is no process $shift-up from $rank"))
        else
            return up(rank,shift,topo)
        end
    end
end
function updown(shift::Int64,topo::Topology)
    updown(topo.rank,shift,topo)
end

struct SimulationTopology
    Nx::Int64
    Nv::Int64
    xdims::Vector{Int64}
    vdims::Vector{Int64}
    topo::Topology
    graphofdims::Matrix{Tuple{Int64, Int64}}
end
function SimulationTopology(Nx::Int64,Nv::Int64,topo::Topology)
    xdims,vdims,graphofdims = distributePhaseSpace(Nx,Nv,topo.dims)
    SimulationTopology(Nx,Nv,xdims,vdims,topo,graphofdims)
end

"""Divides a phase space of size Nx, Nv in the dimensions given by dims.
xdims is the number of x pixels per rank.
vdims is the number of v pixels per rank.
graphofdims is an array of tuples with the dimension of each rank, such that graphofdims[rank+1] gives the local(Nx,Nv) of that rank.
It is better to only use this inside the constructor of SimulationTopology."""
function distributePhaseSpace(Nx,Nv,dims)
    graphofdims = Array{Tuple{Int64,Int64}}(undef, dims...)
    if Nx % dims[2] == 0
        xdims = repeat([Nx÷dims[2]], dims[2])
    else
        xdims = cat(repeat([1+Nx÷dims[2]], dims[2]-1),[ Nx % (1+Nx÷dims[2])],dims=1)
    end
    if Nv % dims[1] == 0
        vdims = repeat([Nv÷dims[1]], dims[1])
    else
        vdims = cat(repeat([1+Nv÷dims[1]], dims[1]-1),[ Nv % (1+Nv÷dims[1])],dims=1)
    end
    for (i,v) in enumerate(vdims)
        for (j,x) in enumerate(xdims)
            graphofdims[i,j] = (v,x)
        end
    end
    xdims, vdims, graphofdims
end

"Returns the corresponding rank and local indexes (j,i) of lattice position (globalj,globali)."
function globalji2rankji(globalj::Int64, globali::Int64,simTopo::SimulationTopology)
    if globalj > +(simTopo.vdims...) || globali > +(simTopo.xdims...)
        throw(ArgumentError("[$globalj, $globali] is outside the grid."))
    end
    globali ÷ length(simTopo.xdims)
    x = (globali-1) ÷ simTopo.xdims[1]
    v = (globalj-1) ÷ simTopo.vdims[1]
    simTopo.topo.graph[v+1,x+1]
    simTopo.topo.graph[v+1,x+1], (globalj - sum(simTopo.vdims[1:v]), globali - sum(simTopo.xdims[1:x]))
end

"""
    localji2globalji(localj::Int64, locali::Int64, rank::Int64, simTopo::SimulationTopology; checklocalbounds::bool = false)

Returns the global indexes corresponding to the local indexes.
"""
function localji2globalji(localj::Int64, locali::Int64, rank::Int64, simTopo::SimulationTopology; checklocalbounds::Bool = false)
    if checklocalbounds && (localj > simTopo.graphofdims[rank+1][1] || locali > simTopo.graphofdims[rank+1][2])
        throw(DomainError(simTopo.graphofdims[rank+1], "Rank $rank has dims $(simTopo.graphofdims[rank+1]).") )
    end
    localj +(simTopo.vdims[1:numberup(rank,simTopo.topo)]...), locali +(simTopo.xdims[1:numberleft(rank,simTopo.topo)]...)
end
localji2globalji(localj::Int64,locali::Int64,simTopo::SimulationTopology) = localji2globalji(localj,locali,simTopo.topo.rank,simTopo)


"""
    localj2globalj(localj::Int64, rank::Int64, simTopo::SimulationTopology; checklocalbounds::bool = false)

Returns the global j index corresponding to the local j index.

"""
function localj2globalj(localj::Int64, rank::Int64, simTopo::SimulationTopology; checklocalbounds::Bool = false)
    if checklocalbounds && (localj > simTopo.graphofdims[rank+1][1] )
        throw(DomainError(simTopo.graphofdims[rank+1], "Rank $rank has dims $(simTopo.graphofdims[rank+1]).") )
    end
    localj +(simTopo.vdims[1:numberup(rank,simTopo.topo)]...)
end
localj2globalj(localj::Int64,simTopo::SimulationTopology) = localj2globalj(localj,simTopo.topo.rank,simTopo; checklocalbounds = false)


"""
    locali2globali(locali::Int64, rank::Int64, simTopo::SimulationTopology; checklocalbounds::Bool = false)

Returns the global i index corresponding to the local i index.
"""
function locali2globali(locali::Int64, rank::Int64, simTopo::SimulationTopology; checklocalbounds::Bool = false)
    if checklocalbounds && (locali > simTopo.graphofdims[rank+1][2] )
        throw(DomainError(simTopo.graphofdims[rank+1], "Rank $rank has dims $(simTopo.graphofdims[rank+1]).") )
    end
    locali +(simTopo.xdims[1:numberleft(rank,simTopo.topo)]...)
end
locali2globali(locali::Int64,simTopo::SimulationTopology) = locali2globali(locali,simTopo.topo.rank,simTopo; checklocalbounds = false)


"""
    initji(rank,simTopo::SimulationTopology)

Returns the initial j and i in the global grid of the given rank.
"""
function initji(rank,simTopo::SimulationTopology)
    if rank >= simTopo.topo.nworkers
        throw(DomainError(rank, "There are only $(simTopo.topo.nworkers) processes.") )
    end
     1 +(simTopo.vdims[1:numberup(rank,simTopo.topo)]...), 1 +(simTopo.xdims[1:numberleft(rank,simTopo.topo)]...)
end
initji(simTopo::SimulationTopology) = initji(simTopo.topo.rank,simTopo::SimulationTopology)

"""
    rangeji(rank,simTopo::SimulationTopology)

Returns two tuples containing the limit j and i of the given rank in the global grid.
"""
function rangeji(rank,simTopo::SimulationTopology)
    inits = initji(rank,simTopo)
    deltas = simTopo.graphofdims[rank+1]
    (inits[1], inits[1] + deltas[1]-1) , (inits[2], inits[2] + deltas[2]-1)
end

# rangeji(2,simTopo)

tuple2range(tup::Tuple{Int64,Int64}) = tup[1]:tup[2]

# initji(3,simTopo)
# simTopo.graphofdims[3]



