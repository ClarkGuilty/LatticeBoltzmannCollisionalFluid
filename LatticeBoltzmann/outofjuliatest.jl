include("LatticeBoltzmann.jl")
# N = 2048
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
# dt = 0.1 * dx/dv
# # dt = 0.2
# G = 0.05
# v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
# x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])
##
sim = Lattice(X_min = -0.5,
 X_max = 0.5,
 Nx = 2048,
 Nv = 2048,
 Nt = 100,
 dt = 0.05,
 V_min=-1.0,
 V_max=1.0,
 G = 0.05,
 grid = gaussian_2d.(
    Float64.(LinRange(-0.5,0.5,2048+1)[1:end-1])',
    Float64.(LinRange(-1.0,1.0,2048+1)[1:end-1])))


@time simulate!(sim)
@time simulate!(sim)
