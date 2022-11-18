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
dt = 0.5 * dx/dv
#v_0 = v_min:dv:v_max
#x_0 = x_min:dx:x_max
v_0 = LinRange(v_min,v_max,N+1)[1:end-1]
x_0 = LinRange(x_min,x_max,N+1)[1:end-1]
G = 0.2729448134597113

ρ=0.25/G
A = 0.01
kj = 0.5
k = 2*(2*π/lx) 
σ = 4π*G*ρ*(kj/k)^2

jeans_init = jeans.(x_0', v_0, σ=0.1,ρ=10,k=4π)
heatmap(jeans_init)
##
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = Nt, dt = dt, V_min=v_min, V_max=v_max,
                grid = copy(jeans_init))

anim = @animate for i in 1:sim.Nt
    heatmap(sim.grid)
    sim.ρ = integrate_lattice!(zeros(size(sim.grid)[2]), sim.grid,sim.dv)
    sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*G)
    sim.a = -num_diff(sim.Φ,1,5,sim.dx)
    rotate_pos!(sim.grid, Vector(v_0))
    rotate_vel!(sim.grid, (-sim.Nv .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% sim.Nv)
end every 1

gif(anim, "/tmp/anim_fps15.gif", fps = 4)
