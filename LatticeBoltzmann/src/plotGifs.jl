include("LatticeBoltzmann.jl")

gr()
#Initializing
N = 2048
Nt = 25
v_min = -1.0
v_max = 1.0
x_min = -0.5
x_max = 0.5
lv = v_max - v_min
lx = x_max - x_min
dv = lv / (N)
dx = lx / (N)
dt = 0.1 * dx/dv
dt = 0.2
G = 1.0
v_0 = Float64.(LinRange(v_min,v_max,N+1)[1:end-1])
x_0 = Float64.(LinRange(x_min,x_max,N+1)[1:end-1])

# ρ=2/G
# A = 0.9
# kj = 0.5
# k = 2*(2*π/lx) 
# σ = 4π*G*ρ*(kj/k)^2

# n = 1
# β = 0.5 
# G = 0.01
# T = lx/lv
# ρ = G^2 / T

ρ = 10
σ = 0.1
A = 0.1
k = 4 * π
# σ = sqrt( G * lx^2 * β^2 * ρ / (π * n^2))
kj = 4 * π * G * ρ / σ^2
# k/kj

jeans_init = jeans.(x_0', v_0, σ=σ,ρ=ρ,k=k,A=A)
heatmap(jeans_init)
##


sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 1,
                dt = dt, V_min=v_min, V_max=v_max, G = 0.00005,
                grid = copy(jeans_init))

heatmap(sim.grid)
@btime simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
heatmap(sim.grid)

##
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 100,
                dt = 0.1, V_min=v_min, V_max=v_max, G = 0.00005,
                grid = copy(jeans_init))




plot(x_0,sim.ρ)
@time simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
@profview simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
heatmap(sim.grid)
##
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 1,
                dt = 0.1, V_min=v_min, V_max=v_max, G = 0.00005,
                grid = copy(jeans_init))



@time anim = @animate for i in 1:100
    heatmap(sim.grid,c = :viridis)
    simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
end every 2
gif(anim, "/tmp/anim_fps15.gif", fps = 10)
##
# @time anim = @animate for i in 1:5
#     heatmap(sim.grid,c = :viridis)
#     integrate_lattice!(sim.ρ,sim.grid,sim.dv)
#     sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*G)
#     sim.a = -num_diff(sim.Φ,1,5,sim.dx)
#     rotate_pos!(sim.grid, Vector(v_0))
#     rotate_vel!(sim.grid, (-sim.Nv .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% sim.Nv)
# end every 1

# @btime integrate_lattice!(sim.ρ,sim.grid,dv)
# @btime sim.ρ = integrate_lattice(sim.grid,sim.dv)



