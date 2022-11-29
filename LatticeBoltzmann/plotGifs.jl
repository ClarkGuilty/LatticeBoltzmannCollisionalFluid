include("LatticeBoltzmann.jl")
##
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

ρ = 1
σ = 0.1
A = 0.1
k = 4 * π
# σ = sqrt( G * lx^2 * β^2 * ρ / (π * n^2))
kj = sqrt(4 * π * G * ρ / σ^2)
# k/kj

jeans_init = jeans.(x_0', v_0, σ=σ,ρ=ρ,k=k,A=A)
heatmap(jeans_init)
##
# sum(sim.grid,dims=2)[:,1]*dv
# @time integrate_lattice!(sim.ρ,sim.grid,sim.dv)
# sum(sim.grid,dims=2)[:,1]
# sim.grid
# sim.ρ

# sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 1,
#                 dt = dt, V_min=v_min, V_max=v_max, G = 0.00005,
#                 grid = copy(jeans_init))

# heatmap(sim.grid)
# @btime simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
# heatmap(sim.grid)

##
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 100,
                dt = 0.1, V_min=v_min, V_max=v_max, G = 0.00005,
                grid = copy(jeans_init))




plot(x_0,sim.ρ)
@time simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
@profview simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
heatmap(sim.grid)
##



##
bullet_init = bullet_cluster.(x_0',v_0;x0=-0.2,x1=0.2,σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=10)
heatmap(bullet_init)
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 1,
                dt = 0.1, V_min=v_min, V_max=v_max, G = 0.5,
                #grid = jeans.(x_0', v_0, σ=σ,ρ=ρ,k=k,A=A))
                grid = bullet_cluster.(x_0',v_0;x0=-0.2,x1=0.2,σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=10))



@time anim = @animate for i in 1:2
    heatmap(sim.grid)
    simulate!(sim,Float64.(x_0),Float64.(v_0);t0=zero(1.0))
end every 2
gif(anim, "/tmp/anim_fps15.gif", fps = 10)
##

sim

ϕ = copy(sim.Φ)
ϕ[1,1] = 0
ϕ
sim.Φ=ϕ
ϕ[1,1] = 10000
sim.Φ
ϕ
