#module LatticeBoltzmann

using FFTW
using Statistics: mean
using Plots
using FiniteDifferences: central_fdm
using Interpolations
using ForwardDiff: gradient
using DiffEqOperators
#using StaticArrays
using Parameters

#include("PoissonSolver.jl")
##PoissonSolver

function λ1(i, n, L)
    if i == 1
        return 0
    end
    -(2*π*fftfreq(n)[i]/L*n)^(-2)
end

λ(n, L=1) = λ1.(1:n,n, L)

function solve_f(rho, L, alpha)
    return real.(ifft(alpha .* fft(rho) .* λ(length(rho),L) ))
    #return ifft(ifftshift(fftshift(fft(rho)) .* λ1(length(rho))))
end

function num_diff(y, degree, approx_order, dx)
    D = CenteredDifference(degree, approx_order, dx, length(y))
    q = PeriodicBC(typeof(y))
    D*(q*y)
end

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
    mass::Float64 = sum(ρ)*dx*dv
    Φ::Vector{Float64} = solve_f(ρ.-mass/Nx,L,4*π*G)
    a::Vector{Float64} = num_diff(Φ,1,5,dx)
end


##
G = 0.001

gaussian_2d(x,y) = gaussian(x,0,0.08) * gaussian(y,0,0.08)

function integrate_lattice!(density::Vector{Float64},grid::Matrix{Float64},dv::Float64)
    for i in 1:size(grid)[1]
        density[i] = 0
        for j in 1:size(grid)[2]
            density[i] += grid[j,i]
        end
    end
    density*dv
end

vel(i,V_min=v_min, dv = dv) = V_min + (1.0*i-0.5)*dv
#vel_i(i,V_min=v_min, dv = dv) = vel(i,V_min=v_min, dv = dv)

function rotate_pos!(arr::Matrix{Float64},v_0::Vector{Float64})
    for i in 1:size(arr)[1]
        circshift!(view(arr,i,:), arr[i,:],(-size(arr)[1] + Int32(round(v_0[i]/dx*dt))) % size(arr)[1])
    end
end

function rotate_vel!(arr::Matrix{Float64}, n::Vector{Int64})
    for i in 1:size(arr)[2]
        #println(n[i])
        circshift!(view(arr,:,i), arr[:,i],n[i])
    end
end
##
gr()
#Initializing
N = 1024
Nt = 1
v_min = -1.0
v_max = 1.0
x_min = -1.0
x_max = 1.0
lv = v_max - v_min
lx = x_max - x_min
dv = lv / (N-1)
dx = lx / (N-1)
dt = 0.1
v_0 = v_min:dv:v_max
x_0 = x_min:dx:x_max
G = 1.0
#gauss_init0 = 4*gaussian_2d.(x_0',v_0)
gauss_init = 4 * exp.((-(x_0 .^2 .+ v_0' .^2)) ./0.08^2)
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = Nt, dt = dt, V_min=v_min, V_max=v_max,
                grid = gauss_init)
#
sim.Nt =500
#sim.dt = 0.1*sim.dx/sim.dv
sim.dt = 0.1
t = 0.0
function simulate!(sim::Lattice, t::Float64, x_0::Vector{Float64}, v_0::Vector{Float64})
    for i in 1:sim.Nt
        sim.ρ = integrate_lattice!(zeros(size(sim.grid)[2]), sim.grid,sim.dv)
        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*G)
        sim.a = -num_diff(sim.Φ,1,5,sim.dx)
        rotate_pos!(sim.grid, v_0)
        rotate_vel!(sim.grid, (-N .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% N)
    end
    t = Nt * sim.dt
end

simulate!(sim,t,Float64.(x_0),Float64.(v_0))
heatmap(sim.grid)

##
plot(sim.ρ)

#end # module
