include("ParallelLatticeBoltzmann.jl")
using JET
using MPI
Plots.default(aspect_ratio=:equal,fmt=:png) 
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

struct topology
    rank::Int64
    nworkers::Int64
    dims::Vector{Int64}
    graph::Matrix{Int64}
end

function topology(comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nworkers = MPI.Comm_size(comm)
    dims = [0,0]
    MPI.Dims_create!(nworkers,2,dims)
    topology(rank,nworkers,dims,collect(reshape(1:nworkers,tuple(dims...))))
end

mpiTopology = topology(comm)

##
#Global init. TODO: update this into args with defaults.
# N = 1024
# Nx = N
# Nv = N
# Nt = 100
# v_min = -1.0
# v_max = 1.0
# x_min = -0.5
# x_max = 0.5
# lv = v_max - v_min
# lx = x_max - x_min
# dv = lv / (Nv)
# dx = lx / (Nx)
# dt = 0.2 * dx/dv
# G = 0.05

# ##Local init
# v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
# x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])
# ##

##

# if rank==0
#     Nrank = (N-oneunit(N) % nworkers)
#     indexinitial = 1
#     indexfinal = Nrank
#     x = x_0[indexinitial:indexfinal]
#     v = v_0[indexinitial:indexfinal]
#     localV_min = v[1]

# end
# localsim = Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = 1,
#     dt = dt, V_min=v_min, V_max=v_max, G = G,
#     grid = gaussian_2d.(x_0',v_0,σx=0.2))



##
# sim = Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = 1,
#                 dt = 0.1, V_min=v_min, V_max=v_max, G = 0.05,
#                 # grid = jeans.(x_0', v_0, σ=σ,ρ=ρ,k=k,A=A))
#                 grid = gaussian_2d.(x_0',v_0))
                # grid = bullet_cluster.(x_0',v_0;x0=-0.2,x1=0.2,σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=10))
# #
# @time anim = @animate for i in 1:20
#     heatmap(sim.grid)
#     simulate!(sim)
# end every 2
# gif(anim, "/tmp/anim_fps15.gif", fps = 1)