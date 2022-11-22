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
Base.IndexStyle(::Type{<:Matrix}) = IndexLinear()
Base.IndexStyle(::Type{<:Matrix}) = IndexCartesian()
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
gaussian_2d(x,v;σx=0.08,σv=0.08,A=40) = gaussian(x,0,σx,A) * gaussian(v,0,σv)
function bullet_cluster(x,v;x0=-0.2,v0=0.0,x1=0.2,v1=0.0,σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=20)
    gaussian(x,x0,σx1,A1) * gaussian(v,v0,σv1) + gaussian(x,x1,σx2,A2) * gaussian(v,v1,σv2)
end
"Integrates the grid matrix with Δv = dv and loads the results on density."
function integrate_lattice!(density::Vector{Float64}, grid::Matrix{Float64}, dv::Float64)
    for i in 1:size(grid,2)
        density[i] = 0
        for j in 1:size(grid,1)
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

@with_kw mutable struct Lattice{T <: AbstractFloat}
    X_min::T = -0.5
    X_max::T = 0.5
    L::T = X_max - X_min
    V_min::T = -1.0
    V_max::T = 1.0
    η::T = V_max - V_min
    Nx::Integer
    Nv::Integer
    Nt::Integer
    dt::T = 0.04
    dx::T = L/Nx
    dv::T = η/Nx
    G::T
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zero(grid)
    ρ::Vector{T} = integrate_lattice(grid,dv)
    mass::T = sum(ρ)*dx
    Φ::Vector{T} = solve_f(ρ.- mass/Nx, L ,4*π*G)
    a::Vector{T} = -num_diff(Φ,1,5,dx)
end


"Velocity initial conditions"
function vel(i;V_min::Float64=-1.0, dv::Float64 = 2/1023)
    V_min + (1.0*(i-1))*dv
end
# vel_i(i,V_min=sim.v_min, dv = sim.dv)
"Drift"
function rotate_pos!(arr::Matrix{Float64},v_0::Vector{Float64})
    for i in 1:size(arr)[1]
        circshift!(view(arr,i,:), arr[i,:],(-size(arr)[1] + Int32(round(v_0[i]/dx*dt))) % size(arr)[1])
    end
    nothing
end

function calculate_new_pos(i::Integer,j::Integer,sim::Lattice)
    # new_i = zero(i)
    # new_j = zero(j)
    Δj = Int(round(sim.a[i]*sim.dt/sim.dv))
    new_j = j + Δj
    if new_j < 1 || new_j > sim.Nv
        # @show i, j, new_j, Int(round(sim.a[i]*sim.dt/sim.dv))
        return -oneunit(new_j), -oneunit(new_j)
    end
    new_i = i + Int(round(vel(new_j;V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))
    mod(new_i-1,sim.Nx)+1, new_j
    # return new_i,new_j
end

function streamingStep!(sim::Lattice)
    phaseTemp = zeros(typeof(sim.grid[1,1]),sim.Nv,sim.Nx)
    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            new_i, new_j = calculate_new_pos(i,j,sim)
            if new_j < 1 || new_j > sim.Nv
                continue
            end
            phaseTemp[new_j,new_i] += sim.grid[j,i]
        end
    end
    sim.grid = phaseTemp
    nothing
end

function streamingStepB!(sim::Lattice)
    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            new_i, new_j = calculate_new_pos(i,j,sim)
            if new_j < 1 || new_j > sim.Nv
                continue
            end
            sim.phaseTemp[new_j,new_i] += sim.grid[j,i]
        end
    end

    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            phase[i][j] = phaseTemp[i][j];
			phaseTemp[i][j] = 0;
        end
    end
    sim.grid = phaseTemp
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
function integrate_steps(sim::Lattice)
    for i in 1:sim.Nt
        integrate_lattice!(sim.ρ, sim.grid, sim.dv)
        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*sim.G)
        sim.a = -num_diff(sim.Φ,1,5,sim.dx)
        streamingStep!(sim)
    end
    nothing
end

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
function simulate!(sim::Lattice; t0::Float64 = 0.0)
    integrate_steps(sim)
    sim.Nt * sim.dt + t0
end

"Runs the simulations"
function simulate!(sim::Lattice, x_0::Vector{Float64}, v_0::Vector{Float64}; t0::Float64 = 0)
    integrate_steps(sim::Lattice, x_0::Vector{Float64}, v_0::Vector{Float64})
    sim.Nt * sim.dt + t0
end

"Goes from linear index to j,i"
function index2ji(index::Integer,velocitySize::Integer)
    index ÷ velocitySize + 1, mod(index-1,velocitySize)+1
end