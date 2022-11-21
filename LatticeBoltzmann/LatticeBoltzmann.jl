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

"Returns the Poisson coefficient 1/λ², takes an int i, array length n and spatial length L."
function λ1(i, n, L)::Float64
    if i == 1
        return zero(1.0)
    end
    -(2*π*fftfreq(n)[i]/L*n)^(-2.0)
end

"Returns and array of λ⁻² of size n and spatial length L."
λ(n, L=1) = λ1.(1:n,n, L)

"Solves Poisson equation for an array rho, representing an spatial length of L and with a coefficient alpha."
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

"Initializes a jeans distribution"
function jeans(x, v; ρ = 0.0001, σ = 0.05, A = 0.9999, k = 4π)
    ρ * exp(- v^2 * 0.5 / σ^2 ) / sqrt(2 * π * σ^2) * (1 + A*cos(k*x))
end

##
#G = 0.001

"2D gaussian"
gaussian_2d(x,y) = gaussian(x,0,0.08) * gaussian(y,0,0.08)
function bullet_cluster(x,v,x0,v0,x1,v1;σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=20)
    gaussian(x,x0,σx1,A1) * gaussian(v,v0,σv1) #+ gaussian(x,x1,σx2,A2) * gaussian(v,v1,σv2)
end
"Integrates the grid matrix with Δv = dv and loads the results on density."
function integrate_lattice!(density::Vector{Float64}, grid::Matrix{Float64}, dv::Float64)
    for i in 1:size(grid)[1]
        density[i] = zero(grid[i,1])
        for j in 1:size(grid)[2]
            density[i] += grid[j,i]
        end
        density[i] *= dv
    end
    nothing
end

"Integrates the grid matrix with Δv = dv and load the results on density."
function integrate_lattice(grid::Matrix{Float64},dv::Float64)
    density = zeros(typeof(grid[end,end]), size(grid)[2])
    integrate_lattice!(density, grid, dv)
    density
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
    dt::Float64 = 0.04
    dx::Float64 = L/Nx
    dv::Float64 = L/Nx
    G::Float64
    grid::Matrix{Float64}
    ρ::Vector{Float64} = integrate_lattice(grid,dv)
    mass::Float64 = sum(ρ)*dx
    Φ::Vector{Float64} = solve_f(ρ.-mass/Nx,L,4*π*G)
    a::Vector{Float64} = num_diff(Φ,1,5,dx)
end


"Velocity initial conditions"
vel(i,V_min=-1.0, dv = 2/1023) = V_min + (1.0*(i-1))*dv
#vel_i(i,V_min=v_min, dv = dv) = vel(i,V_min=v_min, dv = dv)

"Drift"
function rotate_pos!(arr::Matrix{Float64},v_0::Vector{Float64})
    for i in 1:size(arr)[1]
        circshift!(view(arr,i,:), arr[i,:],(-size(arr)[1] + Int32(round(v_0[i]/dx*dt))) % size(arr)[1])
    end
    nothing
end

function calculate_new_pos(i,j)

end

function kick()
    
    nothing
end

"Kick"
function rotate_vel!(arr::Matrix{Float64}, n::Vector{Int64})
    for i in 1:size(arr)[2]
        #println(n[i])
        circshift!(view(arr,:,i), arr[:,i],n[i])
    end
    nothing
end
#

"Time evolves the simulation sim.Nt number of steps."
function integrate_steps(sim::Lattice, x_0::Vector{Float64}, v_0::Vector{Float64})
    for i in 1:sim.Nt
        integrate_lattice!(sim.ρ, sim.grid, sim.dv)
        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*sim.G)
        sim.a = -num_diff(sim.Φ,1,5,sim.dx)
        rotate_pos!(sim.grid, v_0)
        rotate_vel!(sim.grid, (-sim.Nv .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% sim.Nv)
    end
    nothing
end

"Runs the simulations"
function simulate!(sim::Lattice, x_0::Vector{Float64}, v_0::Vector{Float64}; t0::Float64 = 0)
    integrate_steps(sim::Lattice, x_0::Vector{Float64}, v_0::Vector{Float64})
    sim.Nt * sim.dt + t0
end