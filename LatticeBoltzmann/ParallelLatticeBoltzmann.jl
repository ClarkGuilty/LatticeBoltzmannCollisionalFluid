#module LatticeBoltzmann

using FFTW
using Statistics: mean
using Plots
using StatsPlots
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
function num_diff(y, degree, approx_order, dx)::Vector{Float64}
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

# @with_kw mutable struct Lattice{T <: AbstractFloat}
Base.@kwdef mutable struct Lattice{T <: AbstractFloat}
    const X_min::T = -0.5
    const X_max::T = 0.5
    const L::T = X_max - X_min
    const V_min::T = -1.0
    const V_max::T = 1.0
    const η::T = V_max - V_min
    const Nx::Int64
    const Nv::Int64
    const Nt::Int64
    const dx::T = L/Nx
    const dv::T = η/Nx
    const dt::T = 0.2*dx/dv
    const G::T
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zeros(Float64,Nv,Nx)
    new_ji::Vector{Int64} = zeros(Int64,2)
    ρ::Vector{T} = integrate_lattice(grid,dv)
    mass::T = sum(ρ)*dx
    Φ::Vector{T} = solve_f(ρ.- mass/Nx, L ,4*π*G)
    a::Vector{T} = -num_diff(Φ,1,5,dx)
end

Base.@kwdef mutable struct RootLattice{T <: AbstractFloat}
    const global_X_min::T = -0.5
    const global_X_max::T = 0.5
    const L::T = X_max - X_min
    const global_V_min::T = -1.0
    const global_V_max::T = 1.0
    const η::T = V_max - V_min
    const Nx::Int64
    const Nv::Int64
    const Nt::Int64
    const dx::T = L/Nx
    const dv::T = η/Nx
    const dt::T = 0.2*dx/dv
    const G::T
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zeros(Float64,Nv,Nx)
    new_ji::Vector{Int64} = zeros(Int64,2)
    ρ::Vector{T} = integrate_lattice(grid,dv)
    mass::T = sum(ρ)*dx
    Φ::Vector{T} = solve_f(ρ.- mass/Nx, L ,4*π*G)
    a::Vector{T} = -num_diff(Φ,1,5,dx)
end

Base.@kwdef mutable struct ParallelLattice{T <: AbstractFloat}
    const X_min::T
    const X_max::T
    const L::T
    const V_min::T
    const V_max::T
    const Nx::Int64
    const Nv::Int64
    const ΔNx::Int64
    const ΔNv::Int64
    const Nt::Int64
    const dx::T
    const dv::T
    const dt::T
    const G::T
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zeros(Float64,ΔNv,ΔNx)
    new_ji::Vector{Int64} = zeros(Int64,2)
end


"Velocity initial conditions"
function vel(i;V_min::Float64=-1.0, dv::Float64 = 2/1023)
    V_min + (1.0*(i-1))*dv
end
# vel_i(i,V_min=sim.v_min, dv = sim.dv)

function calculate_new_pos(i::Int64,j::Int64,sim::Lattice)::Tuple{Int64,Int64}
    new_j = j + Int(round(sim.a[i]*sim.dt/sim.dv))
    if new_j < 1 || new_j > sim.Nv
        return -oneunit(new_j), -oneunit(new_j)
    end
    mod(i + Int(round(vel(new_j;V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))-oneunit(i),sim.Nx)+oneunit(i), new_j
end

function calculate_new_pos!(i::Int64,j::Int64,sim::Lattice)::Bool
    sim.new_ji[1] = j + Int(round(sim.a[i]*sim.dt/sim.dv))
    if !(oneunit(Int64) < sim.new_ji[1] < sim.Nv)
        return false
    end
    # new_i = i + Int(round(vel(new_j;V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))
    sim.new_ji[2] = mod(i + Int(round(vel(sim.new_ji[1];V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))-oneunit(i),sim.Nx)+oneunit(i)
    true
end

function streamingStep!(sim::Lattice)
    phaseTemp = zeros(Float64,sim.Nv,sim.Nx)
    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            if !calculate_new_pos!(i,j,sim)
                continue
            end
            phaseTemp[sim.new_ji[1],sim.new_ji[2]] += sim.grid[j,i]
        end
    end
    sim.grid = phaseTemp
    nothing
end

function streamingStep!!(sim::Lattice)
    @inbounds for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            if !calculate_new_pos!(i,j,sim)
                continue
            end
            sim.phaseTemp[sim.new_ji[1],sim.new_ji[2]] += sim.grid[j,i]
        end
    end

    @inbounds @simd for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            sim.grid[j,i] = sim.phaseTemp[j,i]
            sim.phaseTemp[j,i] = 0.0

        end
    end
    nothing
end


"Time evolves the simulation sim.Nt number of steps."
function integrate_steps(sim::Lattice)
    for i in 1:sim.Nt
        integrate_lattice!(sim.ρ, sim.grid, sim.dv)
        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*sim.G)
        sim.a = -num_diff(sim.Φ,1,5,sim.dx)
        streamingStep!!(sim)
    end
    nothing
end


"Runs the simulations"
function simulate!(sim::Lattice; t0::Float64 = 0.0)
    integrate_steps(sim)
    sim.Nt * sim.dt + t0
end

"Goes from linear index to j,i"
function index2ji(index::Integer,velocitySize::Integer)
    index ÷ velocitySize + 1, mod(index-1,velocitySize)+1
end
