# include("ParallelLatticeBoltzmann.jl")
# using JET
include("ParallelSugar.jl")
# Plots.default(aspect_ratio=:equal,fmt=:png) 
##
MPI.Init()

comm = MPI.COMM_WORLD


##
# nworkers=3
# dims=[0,0]
# MPI.Dims_create!(nworkers,length(dims),dims)
# testTopo = Topology(3,nworkers,dims,collect(reshape(0:nworkers-1,tuple(dims...))))
# testTopo.graph
testTopo = Topology(comm)
N = 1024
@show testTopo.nworkers
simTopo = SimulationTopology(N,N,testTopo)
@show simTopo.graphofdims
@show simTopo.topo.graph
@show position(simTopo.topo)
##

# numberleft(4,simTopo.topo.)

# @show targetrank, (localj, locali) = globalji2rankji(16, 9,simTopo) 
@show localji2globalji(4,4,simTopo), simTopo.topo.rank



##
# a










