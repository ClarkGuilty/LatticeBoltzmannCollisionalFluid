include("LatticeBoltzmann.jl")

##
gr()
#Initializing
N = 1024
Nt = 100
v_min = -1.0
v_max = 1.0
x_min = -0.5
x_max = 0.5
lv = v_max - v_min
lx = x_max - x_min
dv = lv / (N)
dx = lx / (N)
dt = 0.1*dx/dv
#v_0 = v_min:dv:v_max
#x_0 = x_min:dx:x_max
v_0 = LinRange(v_min,v_max,N+1)[1:end-1]
x_0 = LinRange(x_min,x_max,N+1)[1:end-1]
G = 1.0
#gauss_init0 = 4*gaussian_2d.(x_0',v_0)
#gauss_init = 4 * exp.((-(x_0 .^2 .+ v_0' .^2)) ./0.08^2)
jeans_init = jeans.(x_0', v_0,σ=0.1)


sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = Nt, dt = dt, V_min=v_min, V_max=v_max,
                grid = copy(jeans_init))

@time heatmap(sim.grid)
##

t = 0.0

##

history_M = Array{Float64}(undef, sim.Nt)
history_K = Array{Float64}(undef, sim.Nt)
history_U = Array{Float64}(undef, sim.Nt)

##

@time simulate!(sim,t,Float64.(x_0),
    Float64.(v_0),history_M,history_K,history_U)
@time heatmap(sim.grid)
##


plot(x_0, v_0,sim.grid, st = :heatmap, xaxis = ("Position", (x_min,x_max)),
    yaxis = ("Velocity", (v_min,v_max)),
    c = :viridis)

plot(x_0, v_0,sim.grid, st = :contour,
    xaxis = ("Position"),
    yaxis = ("Velocity"),
    c = :bluesreds)
#marginalhist(sim.grid)
##
p = plot(x_0,sim.ρ, xaxis = ("X", (x_min,x_max), x_min:0.5:x_max ))
plot!(xticks = x_min:0.1:x_max)
theme(:juno)
##end # module

gauss_init = @. 4 * exp.((-(x_0 .^2 .+ v_0' .^2)) ./0.08^2)
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = Nt, dt = dt, V_min=v_min, V_max=v_max,
                grid = copy(gauss_init))
#
sim.Nt =20
#sim.dt = 0.1*sim.dx/sim.dv
sim.dt = 0.1
t = 0.0
