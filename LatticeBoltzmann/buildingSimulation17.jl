@show 00
include("ParallelSugar.jl")
include("ParallelLatticeBoltzmann17.jl")
include("ParallelLatticeUtilities.jl")
##
@show 123


DENSITYTAG = 3541
##
MPI.Init()
comm = MPI.COMM_WORLD
if MPI.Comm_rank(comm) == 0
    import Plots
    Plots.gr()
    Plots.default(aspect_ratio=:equal,fmt=:png) 
end
##
@show MPI.Comm_rank(comm)
const N = 1024
const Nx = N
const Nv = N
const Nt = 25
const v_min = -1.0
const v_max = 1.0
const x_min = -0.5
const x_max = 0.5
const lv = v_max - v_min
const lx = x_max - x_min
const dv = lv / (Nv)
const dx = lx / (Nx)
const dt = 0.1 * dx/dv
# dt = 0.2
const G = 0.05
v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])


##
# rank = 5
# nworkers=6
# dims=[0,0]
# MPI.Dims_create!(nworkers,length(dims),dims)
# testTopo = Topology(rank,nworkers,dims,collect(reshape(0:nworkers-1,tuple(dims...))))
# testTopo.graph


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
                dt = dt, G = G, grid = gaussian_2d.(x_local',v_local))
#

localj=1
globalj = localj2globalj(localj,simTopo)
# vel(localj;V_min=localLattice.V_min, dv=localLattice.dv) #This works, it tested it.


##
MPI.Bcast!(localLattice.a, 0, comm)
integrate_lattice!(localLattice.ρ, localLattice.grid, localLattice.dv)

if simTopo.topo.rank == 0
    localLattice.a = sim.a
end

# syncronizedensity(localLattice, sim, simTopo,comm)


