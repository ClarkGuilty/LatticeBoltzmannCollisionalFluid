# include("ParallelLatticeBoltzmann.jl")
# using JET
using MPI
# Plots.default(aspect_ratio=:equal,fmt=:png) 
##
MPI.Init()

comm = MPI.COMM_WORLD
# rank = MPI.Comm_rank(comm)
# nworkers = MPI.Comm_size(comm)
# ##
# nworkers = 10
# dims = [0,0] #Initialization. 0 means you accept what MPI thinks is best.
# MPI.Dims_create!(nworkers,2,dims)
# topologyGrid = reshape(1:nworkers,tuple(dims...))
##

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
position(rank::Int64, topo::Topology)::Int64 = topo.graph[rank]
"Returns the position in the Topology of the process calling it."
position(topo::Topology)::Int64 = topo.graph[topo.rank]

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

topo = Topology(comm)
@show topo.graph
# @show topo = topo
@show topo.rank, leftright(topo.rank,1,topo)
@show topo.rank, leftright(topo.rank,-1,topo)
@show topo.rank, updown(topo.rank,1,topo)
@show topo.rank, updown(topo.rank,-1,topo)

a
##
dims = [6,3]
testTopo = Topology(3,18,dims,collect(reshape(0:18-1,tuple(dims...))))
testTopo.graph
leftright(-1,testTopo)
leftright(1,testTopo)

numberup(18,testTopo)
numberdown(12,testTopo)


# down(5,2,testTopo)
updown(1,-1,testTopo)
# pos = 2

a