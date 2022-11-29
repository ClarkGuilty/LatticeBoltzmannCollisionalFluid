# include("LatticeBoltzmann.jl")
##
using Plots
unicodeplots()
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

# heatmap(jeans_init)
plot(x_0, label="123")
p = plot!(v_0, label="323")
savefig(p,"fml.png")
# gui(p)
