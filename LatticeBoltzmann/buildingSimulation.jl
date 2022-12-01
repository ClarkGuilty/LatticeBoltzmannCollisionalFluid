include("ParallelLatticeBoltzmann.jl")
include("ParallelLatticeUtilities.jl")
# using BenchmarkTools
using Profile, ProfileSVG
##
const DENSITYTAG::Int64 = 3541
const GRIDTAG::Int64 = 1254
MPI.Init()
comm = MPI.COMM_WORLD
# if MPI.Comm_rank(comm) == 0
    # import Plots
    # Plots.gr()
    # Plots.default(aspect_ratio=:equal,fmt=:png) 
    # true
# end
##
const N::Int64 = 2048
const Nx::Int64 = N
const Nv::Int64 = N
const Nt::Int64 = 100
const v_min::Float64 = -1.0
const v_max::Float64 = 1.0
const x_min::Float64 = -0.5
const x_max::Float64 = 0.5
const lv::Float64 = v_max - v_min
const lx::Float64 = x_max - x_min
const dv::Float64 = lv / (Nv)
const dx::Float64 = lx / (Nx)
const dt::Float64 = 0.1 * dx/dv
const G::Float64 = 0.1
v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])


##
# rank = 1
# nworkers=6
# dims=[0,0]
# MPI.Dims_create!(nworkers,length(dims),dims)
# otherworkers = collect(0:nworkers-1)[0:nworkers-1 .!= rank]
# testTopo = Topology(rank,nworkers,dims,collect(reshape(0:nworkers-1,tuple(dims...))),otherworkers)
# testTopo.graph
# testTopo.otherworkers
#
testTopo = Topology(comm)
simTopo = SimulationTopology(N,N,testTopo)

sim = simTopo.topo.rank == 0 ? Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = Nt,
    dt = dt, V_min=v_min, V_max=v_max, G = G,
    grid = gaussian_2d.(x_0',v_0)) : nothing
#

# if simTopo.topo.rank == 0
#     Plots.plot(sim.ρ,label="t=0")
#     simulate!(sim)
#     Plots.plot!(sim.ρ,label="Serial")
#     Plots.title!("ρ at $(Nt)")
# end
#
# sim = simTopo.topo.rank == 0 ? Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = Nt,
#     dt = dt, V_min=v_min, V_max=v_max, G = G,
#     grid = gaussian_2d.(x_0',v_0)) : nothing

x_local = x_0[initji(simTopo)[2]:initji(simTopo)[2]+simTopo.graphofdims[simTopo.topo.rank+1][2]-1]
v_local = v_0[initji(simTopo)[1]:initji(simTopo)[1]+simTopo.graphofdims[simTopo.topo.rank+1][1]-1]
localLattice = ParallelLattice(X_min = x_local[1], X_max = x_local[end],L=lx,V_min = v_local[1], V_max = v_local[end],
                Nx = N, Nv = N, ΔNx = simTopo.graphofdims[simTopo.topo.rank+1][2],
                ΔNv = simTopo.graphofdims[simTopo.topo.rank+1][1], Nt = Nt, dx = dx, dv = dv,
                dt = dt, G = G, grid = gaussian_2d.(x_local',v_local),
                sendMessages = [GridMessage([],[],i,simTopo.topo.rank,[]) for i in 0:simTopo.topo.nworkers-1 ],
                recvMessages = [GridMessage([],[],i,simTopo.topo.rank,[]) for i in 0:simTopo.topo.nworkers-1 ])
                # recvMessages = [Vector{ActualMessage}(undef, 1) for i in 1:simTopo.topo.nworkers] )
                # recvMessages = [GridMessage([],i,simTopo.topo.rank,[]) for i in 0:simTopo.topo.nworkers-1 ],
                # sendMessages = [GridMessage([],i,simTopo.topo.rank,[]) for i in 0:simTopo.topo.nworkers-1 ])
#
if simTopo.topo.rank == 0
    localLattice.a = sim.a
end


##
# @show "I got to the streaming step" 
# @show localLattice,simTopo,comm
# parallel_streamingStep!!(localLattice,simTopo)
# multinodeStreamingStep!(localLattice,simTopo,comm)
# @profile parallelIntegrate_steps!(localLattice,sim,simTopo)
@time parallelIntegrate_steps!(localLattice,sim,simTopo)

# ProfileSVG.save("prof"*string(simTopo.topo.rank)*".svg";width=5000)
# @show " im done $(simTopo.topo.rank)"

# @benchmark(parallelIntegrate_steps!(localLattice,sim,simTopo),
#            teardown=MPI.Barrier(comm), # Note - includes at least one MPI.Barrier()
#            seconds=60,
#            samples=5,
#            evals=1)
