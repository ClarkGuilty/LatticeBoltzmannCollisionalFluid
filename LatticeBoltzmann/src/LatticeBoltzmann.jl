#module LatticeBoltzmann

using FFTW
using Statistics: mean
using Plots, StatsPlots
using FiniteDifferences: central_fdm
using Interpolations
using ForwardDiff: gradient
using DiffEqOperators
using Parameters
using BenchmarkTools

#include("PoissonSolver.jl")
##PoissonSolver

"Returns the Poisson coefficient 1/λ², takes a int i, array length n and spatial length L."
function λ1(i, n, L)
    if i == 1
        return 0
    end
    -(2*π*fftfreq(n)[i]/L*n)^(-2)
end

"Returns and array of λ⁻² of array length n and spatial length L."
λ(n, L=1) = λ1.(1:n,n, L)

"Solves Poisson equation for an array rho, rerpresenting an spatial length of L and with a coefficient alpha."
function solve_f(rho, L, alpha)
    return real.(ifft(alpha .* fft(rho) .* λ(length(rho),L) ))
    #return ifft(ifftshift(fftshift(fft(rho)) .* λ1(length(rho))))
end

"degree derivative of y. Central difference scheme with order approx_order and Δx=dx."
function num_diff(y, degree, approx_order, dx)
    D = CenteredDifference(degree, approx_order, dx, length(y))
    q = PeriodicBC(typeof(y))
    D*(q*y)
end

"Gaussian initialization. μ mean, σ standard deviation, and A is the amplitude."
function gaussian(x, μ=0,σ=1, A=1)
    A * exp(-((x - μ) / σ)^2)
end

@with_kw mutable struct Lattice
    X_min::Float64 = -0.5
    X_max::Float64 = 0.5
    L::Float64 = X_max - X_min
    V_min::Float64 = -1
    V_max::Float64 = 1
    η::Float64 = V_max - V_min
    Nx::Integer
    Nv::Integer
    Nt::Integer
    dt::Float64 = 0.01
    dx::Float64 = L/Nx
    dv::Float64 = L/Nx
    grid::Matrix{Float64}
    ρ::Vector{Float64} = integrate_lattice!(zeros(size(grid)[2]), grid,dv)
    mass::Float64 = sum(ρ)*dx
    Φ::Vector{Float64} = solve_f(ρ.-mass/Nx,L,4*π*G)
    a::Vector{Float64} = num_diff(Φ,1,5,dx)
end


##
G = 0.001

"2D gaussian"
gaussian_2d(x,y) = gaussian(x,0,0.08) * gaussian(y,0,0.08)

"Integrates the grid matrix with Δv = dv and load the results on density."
function integrate_lattice!(density::Vector{Float64},grid::Matrix{Float64},dv::Float64)
    for i in 1:size(grid)[1]
        density[i] = 0
        for j in 1:size(grid)[2]
            density[i] += grid[j,i]
        end
    end
    density = density*dv
end

"Integrates the grid matrix with Δv = dv and load the results on density, returns the total mass, the kinetic energy and potential energy."
function integrate_lattice!(density::Vector{Float64},grid::Matrix{Float64},dx::Float64,dv::Float64,v::Vector{Float64},phi::Vector{Float64})
    total_mass = 0
    total_K_e = 0
    total_U_e = 0
    for i in 1:size(grid)[1]
        density[i] = 0
        for j in 1:size(grid)[2]
            density[i] += grid[j,i]
            total_K_e += grid[j,i]*v[j]^2
        end
        total_mass += density[i]
        total_U_e += phi[i]*density[i]*dv
    end
    density = density .* dv
    total_mass*dx*dv, 0.5*total_K_e*dx*dv, 0.5*total_U_e*dx
end

"Velocity initial conditions"
vel(i,V_min=-1, dv = 2/1023) = V_min + (1.0*(i-1))*dv
#vel_i(i,V_min=v_min, dv = dv) = vel(i,V_min=v_min, dv = dv)

"Drift"
function rotate_pos!(arr::Matrix{Float64},v_0::Vector{Float64})
    for i in 1:size(arr)[1]
        circshift!(view(arr,i,:), arr[i,:],(-size(arr)[1] + Int32(round(v_0[i]/dx*dt))) % size(arr)[1])
    end
end

"Kick"
function rotate_vel!(arr::Matrix{Float64}, n::Vector{Int64})
    for i in 1:size(arr)[2]
        #println(n[i])
        circshift!(view(arr,:,i), arr[:,i],n[i])
    end
end
##
#gr()
#Initializing
N = 1024
Nt = 10
v_min = -1.0
v_max = 1.0
x_min = -1.0
x_max = 1.0
lv = v_max - v_min
lx = x_max - x_min
dv = lv / (N-1)
dx = lx / (N-1)
dt = 0.1
#v_0 = v_min:dv:v_max
#x_0 = x_min:dx:x_max
v_0 = LinRange(v_min,v_max,N+1)[1:end-1]
x_0 = LinRange(x_min,x_max,N+1)[1:end-1]
G = 1.0
#gauss_init0 = 4*gaussian_2d.(x_0',v_0)
gauss_init = 4 * exp.((-(x_0 .^2 .+ v_0' .^2)) ./0.08^2)
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = Nt, dt = dt, V_min=v_min, V_max=v_max,
                grid = copy(gauss_init))
#
sim.Nt =200
#sim.dt = 0.1*sim.dx/sim.dv
sim.dt = 0.1
t = 0.0


"Runs the simulations"
function simulate!(sim::Lattice, t::Float64, x_0::Vector{Float64}, v_0::Vector{Float64},history_M,history_K, history_U)
    for i in 1:sim.Nt
#        sim.ρ = integrate_lattice!(zeros(size(sim.grid)[2]), sim.grid,sim.dv)
        (history_M[i], history_K[i], history_U[i]) = integrate_lattice!(sim.ρ, sim.grid, sim.dx,sim.dv, Vector(v_0),sim.Φ)
        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*G)
        sim.a = -num_diff(sim.Φ,1,5,sim.dx)
        rotate_pos!(sim.grid, v_0)
        rotate_vel!(sim.grid, (-N .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% N)
    end
    t = sim.Nt * sim.dt
end

history_M = Array{Float64}(undef, sim.Nt)
history_K = Array{Float64}(undef, sim.Nt)
history_U = Array{Float64}(undef, sim.Nt)

@time simulate!(sim,t,Float64.(x_0),Float64.(v_0),history_M,history_K,history_U)
##
plot(x_0, v_0,sim.grid, st = :contour, xaxis = ("Position", (x_min/3,x_max/3), x_min:0.25:x_max ),
    yaxis = ("Velocity", (v_min/3,v_max/3), v_min:0.25:v_max ),
    c = :bluesreds)
#marginalhist(sim.grid)
##
p = plot(x_0,sim.ρ, xaxis = ("my label", (x_min,x_max), x_min:0.5:x_max ))
plot!(xticks = x_min:0.1:x_max)
theme(:juno)
##end # module

gauss_init = 4 * exp.((-(x_0 .^2 .+ v_0' .^2)) ./0.08^2)
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = Nt, dt = dt, V_min=v_min, V_max=v_max,
                grid = copy(gauss_init))
#
sim.Nt =20
#sim.dt = 0.1*sim.dx/sim.dv
sim.dt = 0.1
t = 0.0
anim = @animate for i in 1:sim.Nt
    plot!(x_0, v_0,sim.grid, st = :contour, xaxis = ("Position", (x_min/3,x_max/3), x_min:0.25:x_max ),
         yaxis = ("Velocity", (v_min/3,v_max/3), v_min:0.25:v_max ),
        c = :bluesreds)
    sim.ρ = integrate_lattice!(zeros(size(sim.grid)[2]), sim.grid,sim.dv)
    sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*G)
    sim.a = -num_diff(sim.Φ,1,5,sim.dx)
    rotate_pos!(sim.grid, Vector(v_0))
    rotate_vel!(sim.grid, (-N .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% N)
end every 1

gif(anim, "anim_fps15.gif", fps = 4)
