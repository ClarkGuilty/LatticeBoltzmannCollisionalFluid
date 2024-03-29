#module LatticeBoltzmann
using FFTW
using Statistics: mean
# using Plots, StatsPlots
using FiniteDifferences: central_fdm
using Interpolations
using ForwardDiff: gradient
using DiffEqOperators
# using Parameters
# using BenchmarkTools

# Base.IndexStyle(::Type{<:Matrix}) = IndexLinear()
# Base.IndexStyle(::Type{<:Matrix}) = IndexCartesian()

"""
    integrate_lattice(grid::Matrix{Float64},dv::Float64)

Integrates the grid matrix with Δv = dv and load the results on density.
"""
function integrate_lattice(grid::Matrix{Float64},dv::Float64)
    density = zeros(typeof(grid[end,end]), size(grid)[2])
    integrate_lattice!(density, grid, dv)
    density
end

"""
    integrate_lattice!(density::Vector{Float64}, grid::Matrix{Float64}, dv::Float64)

Integrates the grid matrix with Δv = dv and loads the results on density.
"""
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

Base.@kwdef mutable struct Lattice{T <: AbstractFloat}
    X_min::T = -0.5
    X_max::T = 0.5
    L::T = X_max - X_min
    V_min::T = -1.0
    V_max::T = 1.0
    η::T = V_max - V_min
    Nx::Int64
    Nv::Int64
    Nt::Int64
    dx::T = L/Nx
    dv::T = η/Nx
    dt::T = 0.2*dx/dv
    G::T
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zeros(Float64,Nv,Nx)
    new_ji::Vector{Int64} = zeros(Int64,2)
    ρ::Vector{T} = integrate_lattice(grid,dv)
    mass::T = sum(ρ)*dx
    Φ::Vector{T} = solve_f(ρ.- mass/Nx, L ,4*π*G)
    a::Vector{T} = -num_diff(Φ,1,5,dx)
end

Base.@kwdef mutable struct ParallelLattice{T <: AbstractFloat}
    X_min::T
    X_max::T
    L::T
    V_min::T
    V_max::T
    Nx::Int64
    Nv::Int64
    ΔNx::Int64
    ΔNv::Int64
    Nt::Int64
    dx::T
    dv::T
    dt::T
    G::T
    # const x_0::Vector{T}
    # const v_0::Vector{T}
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zeros(Float64,ΔNv,ΔNx)
    new_localji::Vector{Int64} = zeros(Int64,2)
    new_globalji::Vector{Int64} = zeros(Int64,2)
    ρ::Vector{T} = integrate_lattice(grid,dv)
    a::Vector{T} = zeros(Float64,Nx)
end


"""
    gaussian(x, μ=0,σ=1, A=1)

Gaussian initialization. μ mean, σ standard deviation, and A is the amplitude.
"""
gaussian(x, μ=0,σ=1, A=1) = A * exp(-((x - μ) / σ)^2)
"2D gaussian"
gaussian_2d(x,v;σx=0.08,σv=0.08,A=40) = gaussian(x,0,σx,A) * gaussian(v,0,σv)
function bullet_cluster(x,v;x0=-0.2,v0=0.0,x1=0.2,v1=0.0,σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=20)
    gaussian(x,x0,σx1,A1) * gaussian(v,v0,σv1) + gaussian(x,x1,σx2,A2) * gaussian(v,v1,σv2)
end

"""
    λ1(i, n, L)::Float64

Returns the Poisson coefficient 1/λ², takes an int i, array length n and spatial length L.
"""
function λ1(i, n, L)::Float64
    if i == 1
        return zero(1.0)
    end
    -(2*π*fftfreq(n)[i]/L*n)^(-2.0)
end

"""
    λ(n, L=1)

Returns and array of λ⁻² of size n and spatial length L.
"""
λ(n, L=1) = λ1.(1:n,n, L)

"""
    solve_f(rho, L, alpha)

Solves Poisson equation for an array rho, representing an spatial length of L and with a coefficient alpha.
"""
function solve_f(rho, L, alpha)
    return real.(ifft(alpha .* fft(rho) .* λ(length(rho),L) ))
    #return ifft(ifftshift(fftshift(fft(rho)) .* λ1(length(rho))))
end

"""
    num_diff(y, degree, approx_order, dx)::Vector{Float64}

degree derivative of y. Central difference scheme with order approx_order and Δx=dx.
"""
function num_diff(y, degree, approx_order, dx)::Vector{Float64}
    D = CenteredDifference(degree, approx_order, dx, length(y))
    q = PeriodicBC(typeof(y))
    D*(q*y)
end

"""
    vel(i;V_min::Float64=-1.0, dv::Float64 = 2/1023)

Velocity initial conditions
"""
function vel(i;V_min::Float64=-1.0, dv::Float64 = 2/1023)
    V_min + (1.0*(i-1))*dv
end
# vel_i(i,V_min=sim.v_min, dv = sim.dv)


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
    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            if !calculate_new_pos!(i,j,sim)
                continue
            end
            sim.phaseTemp[sim.new_ji[1],sim.new_ji[2]] += sim.grid[j,i]
        end
    end

    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            sim.grid[j,i] = sim.phaseTemp[j,i]
            sim.phaseTemp[j,i] = 0.0

        end
    end
    nothing
end


"""
    integrate_steps(sim::Lattice)

Time evolves the simulation sim.Nt number of steps.
"""
function integrate_steps(sim::Lattice)
    for i in 1:sim.Nt
        integrate_lattice!(sim.ρ, sim.grid, sim.dv)
        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*sim.G)
        sim.a = -num_diff(sim.Φ,1,5,sim.dx)
        streamingStep!!(sim)
    end
    nothing
end


"""
    parallelCalculate_new_pos!(i::Int64,j::Int64,localLattice::ParallelLattice, simTopo::SimulationTopology)::Bool

TBW
"""
function parallelCalculate_new_pos!(i::Int64,j::Int64,localLattice::ParallelLattice, simTopo::SimulationTopology)::Bool
    localLattice.new_localji[1] = j + Int(round(
        localLattice.a[locali2globali(i,simTopo)]*localLattice.dt/localLattice.dv))
    
    if !(oneunit(Int64) < localj2globalj(localLattice.new_ji[1],simTopo) < localLattice.Nv)
        return false
    end
    # new_i = i + Int(round(vel(new_j;V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))
    localLattice.new_localji[2] = mod(i + Int(round(vel(
        localLattice.new_ji[1];V_min=localLattice.V_min,dv=localLattice.dv)*localLattice.dt/localLattice.dx))-oneunit(i),localLattice.Nx)+oneunit(i)
    true
end

"""
    parallel_streamingStep!!(sim::Lattice)

TBW
"""
function parallel_streamingStep!!(sim::Lattice)
    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            if !calculate_new_pos!(i,j,sim)
                continue
            end
            sim.phaseTemp[sim.new_ji[1],sim.new_ji[2]] += sim.grid[j,i]
        end
    end

    for i in 1:size(sim.grid,2)
        for j in 1:size(sim.grid,1)
            sim.grid[j,i] = sim.phaseTemp[j,i]
            sim.phaseTemp[j,i] = 0.0

        end
    end
    nothing
end

"""
    parallel_integrate_steps(sim::Lattice)

Time evolves the simulation sim.Nt number of steps.
"""
function parallelIntegrate_steps(localLattice::ParallelLattice,sim::Union{Lattice,Nothing})
    for i in 1:sim.Nt
        Integrate_lattice!(sim.ρ, sim.grid, sim.dv)
        syncronizedensity(localLattice, sim, simTopo,comm)

        sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*sim.G) #parallel version
        sim.a = -num_diff(sim.Φ,1,5,sim.dx) #parallel version
        MPI.Bcast!(localLattice.a, 0, comm)
        parallelStreamingStep!!(sim)


    end
    nothing
end


"""
    simulate!(sim::Lattice; t0::Float64 = 0.0)

Runs the simulations
"""
function simulate!(sim::Lattice; t0::Float64 = 0.0)
    integrate_steps(sim)
    sim.Nt * sim.dt + t0
end

"""
    index2ji(index::Integer,velocitySize::Integer)

Goes from linear index to j,i
"""
function index2ji(index::Integer,velocitySize::Integer)
    index ÷ velocitySize + 1, mod(index-1,velocitySize)+1
end

