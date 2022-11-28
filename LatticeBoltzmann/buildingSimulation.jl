include("ParallelSugar.jl")
include("ParallelLatticeBoltzmann.jl")

# using Plots
# Plots.default(aspect_ratio=:equal,fmt=:png) 



const DENSITYTAG = 3541
##
MPI.Init()
comm = MPI.COMM_WORLD
##
N = 1024
Nx = N
Nv = N
Nt = 25
v_min = -1.0
v_max = 1.0
x_min = -0.5
x_max = 0.5
lv = v_max - v_min
lx = x_max - x_min
dv = lv / (Nv)
dx = lx / (Nx)
dt = 0.1 * dx/dv
# dt = 0.2
G = 1.0
v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])


##
# rank = 0
# nworkers=6
# dims=[0,0]
# MPI.Dims_create!(nworkers,length(dims),dims)
# testTopo = Topology(rank,nworkers,dims,collect(reshape(0:nworkers-1,tuple(dims...))))
# testTopo.graph
testTopo = Topology(comm)
@show testTopo.nworkers
simTopo = SimulationTopology(N,N,testTopo)
simTopo.graphofdims

x_local = x_0[initji(simTopo)[2]:initji(simTopo)[2]+simTopo.graphofdims[simTopo.topo.rank+1][2]-1]
v_local = v_0[initji(simTopo)[1]:initji(simTopo)[1]+simTopo.graphofdims[simTopo.topo.rank+1][1]-1]
localLattice = ParallelLattice(X_min = x_local[1], X_max = x_local[end],L=lx,V_min = v_local[1], V_max = v_local[end],
                Nx = N, Nv = N, ΔNx = simTopo.graphofdims[simTopo.topo.rank+1][2],
                ΔNv = simTopo.graphofdims[simTopo.topo.rank+1][1], Nt = Nt, dx = dx, dv = dv,
                dt = dt, G = G, grid = gaussian_2d.(x_local',v_local))
#
# heatmap(gaussian_2d.(x_local',v_local),aspect_ratio=:equal,yflip=true,clims=(0,40))
# plot(localLattice.ρ)

##
##
integrate_lattice!(localLattice.ρ, localLattice.grid, localLattice.dv)

function send2root(arr::Vector{Float64},tag::Int64,comm::MPI.Comm)
    MPI.Isend(arr, comm; dest=0, tag)
end

function recvfromall(tag::Int64,simTopo::SimulationTopology,comm::MPI.Comm)
    sreqs_workers = Array{MPI.Request}(undef,simTopo.topo.nworkers-1)
    buffers = Vector{Vector{Float64}}(undef,simTopo.topo.nworkers-1)
    for rank in 1:simTopo.topo.nworkers-1
        @show rank
        buffers[rank] = zeros(Float64,simTopo.graphofdims[rank+1][2])
        sreqs_workers[rank] = MPI.Irecv!(buffers[rank], comm; source=rank, tag=tag)
    end
    @show sreqs_workers
    stats = MPI.Waitall(sreqs_workers)
    buffers, stats
end

if simTopo.topo.rank != 0
    req = send2root(localLattice.ρ,DENSITYTAG,comm)
    MPI.Waitall([req])
end
##
if simTopo.topo.rank == 0
    # sim = Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = Nt,
    #             dt = dt, V_min=v_min, V_max=v_max, G = G,
    #             grid = gaussian_2d.(x_0',v_0))
    buffers, stats = recvfromall(DENSITYTAG,simTopo,comm)
    # @show buffers
    # @show stats
    for buffer in buffers
        @show buffer
    end
end

density = zeros(Float64,Nx)
density[tuple2range(rangeji(3,simTopo)[2])] += 
x_0[tuple2range(rangeji(3,simTopo)[2])]

# for worker in 1:simTopo.topo.nworkers-1
#     @show worker
# end

# buffers = Vector{Vector{Float64}}(undef,nworkers-1)
# buffers[1] = zeros(Float64,length(localLattice.ρ))