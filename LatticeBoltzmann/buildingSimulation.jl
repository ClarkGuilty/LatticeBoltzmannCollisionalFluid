include("ParallelLatticeBoltzmann.jl")
include("ParallelLatticeUtilities.jl")
##



const DENSITYTAG::Int64 = 3541
##
MPI.Init()
comm = MPI.COMM_WORLD
if MPI.Comm_rank(comm) == 0
    import Plots
    Plots.gr()
    Plots.default(aspect_ratio=:equal,fmt=:png) 
end
##
const N::Int64 = 1024
const Nx::Int64 = N
const Nv::Int64 = N
const Nt::Int64 = 25
const v_min::Float64 = -1.0
const v_max::Float64 = 1.0
const x_min::Float64 = -0.5
const x_max::Float64 = 0.5
const lv::Float64 = v_max - v_min
const lx::Float64 = x_max - x_min
const dv::Float64 = lv / (Nv)
const dx::Float64 = lx / (Nx)
const dt::Float64 = 0.1 * dx/dv
const G::Float64 = 0.05
v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])


##
rank = 5
nworkers=6
dims=[0,0]
MPI.Dims_create!(nworkers,length(dims),dims)
testTopo = Topology(rank,nworkers,dims,collect(reshape(0:nworkers-1,tuple(dims...))))
testTopo.graph
##

testTopo = Topology(comm)
# @show testTopo.nworkers
simTopo = SimulationTopology(N,N,testTopo)
simTopo.graphofdims

sim = simTopo.topo.rank == 0 ? Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = Nt,
    dt = dt, V_min=v_min, V_max=v_max, G = G,
    grid = gaussian_2d.(x_0',v_0)) : nothing


x_local = x_0[initji(simTopo)[2]:initji(simTopo)[2]+simTopo.graphofdims[simTopo.topo.rank+1][2]-1]
v_local = v_0[initji(simTopo)[1]:initji(simTopo)[1]+simTopo.graphofdims[simTopo.topo.rank+1][1]-1]
localLattice = ParallelLattice(X_min = x_local[1], X_max = x_local[end],L=lx,V_min = v_local[1], V_max = v_local[end],
                Nx = N, Nv = N, ΔNx = simTopo.graphofdims[simTopo.topo.rank+1][2],
                ΔNv = simTopo.graphofdims[simTopo.topo.rank+1][1], Nt = Nt, dx = dx, dv = dv,
                dt = dt, G = G, grid = gaussian_2d.(x_local',v_local),
                recvMessages = [GridMessage([],i,simTopo.topo.rank,[]) for i in 0:simTopo.topo.nworkers-1 ],
                sendMessages = [GridMessage([],i,simTopo.topo.rank,[]) for i in 0:simTopo.topo.nworkers-1 ])
#
if simTopo.topo.rank == 0
    localLattice.a = sim.a
end


localj=1
globalj = localj2globalj(localj,simTopo)
# vel(localj;V_min=localLattice.V_min, dv=localLattice.dv) #This works, it tested it.


##
# MPI.Bcast!(localLattice.a, 0, comm)
# integrate_lattice!(localLattice.ρ, localLattice.grid, localLattice.dv)


# syncronizedensity(localLattice, sim, simTopo,comm)

parallelCalculate_new_pos!(10,11,localLattice, simTopo)
new_globalji = tuple2vector(localji2globalji(localLattice.new_localji[1],localLattice.new_localji[2] ,simTopo))
localji2globaljivect(123,-120 ,simTopo)
rank, localLattice.new_localji = globalji2rankjivect(new_globalji[1]+12,new_globalji[2],simTopo)

localLattice.new_localji
temp[2]
localLattice.new_localji
localLattice.new_globalji
localLattice.new_tempji
localLattice.new_target















