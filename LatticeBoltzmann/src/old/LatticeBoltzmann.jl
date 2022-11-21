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
function λ1(i, n, L)::AbstractFloat
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


"Integrates the grid matrix with Δv = dv and load the results on density."
function integrate_lattice!(output::Vector{Float64}, grid::Matrix{Float64}, dv::Float64)
    for i in 1:size(grid)[1]
        output[i] = zero(grid[i,1])
        for j in 1:size(grid)[2]
            output[i] += grid[j,i]
        end
        output[i] *= dv
    end
    nothing
end


"Integrates the grid matrix with Δv = dv and load the results on density."
function integrate_lattice(grid::Matrix{Float64},dv::Float64)
    density = zeros(typeof(grid[end,end]), size(grid)[2])
    for i in 1:size(density)[1]
        density[i] = zero(density[i])
        for j in 1:size(grid)[2]
            density[i] += grid[j,i]
        end
    end
    density = density .* dv
end

"Integrates the grid matrix with Δv = dv and load the results on density, returns the total mass, the kinetic energy and potential energy."
function integrate_lattice!(density::Vector{Float64},
    grid::Matrix{Float64},dx::Float64,dv::Float64,
    v::Vector{Float64},phi::Vector{Float64})
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

"Kick"
function rotate_vel!(arr::Matrix{Float64}, n::Vector{Int64})
    for i in 1:size(arr)[2]
        #println(n[i])
        circshift!(view(arr,:,i), arr[:,i],n[i])
    end
    nothing
end
#

"Runs the simulations"
function simulate_store_energies!(sim::Lattice, t::Float64, x_0::Vector{Float64},
    v_0::Vector{Float64},history_M::Array{Float64},
    history_K::Array{Float64}, history_U::Array{Float64})
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