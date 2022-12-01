#module LatticeBoltzmann
using FFTW
using Statistics: mean
# using Plots, StatsPlots
using FiniteDifferences: central_fdm
using Interpolations
using ForwardDiff: gradient
using DiffEqOperators
using StaticArrays

# using Parameters
# using BenchmarkTools

include("ParallelSugar.jl")

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


struct GridMessage
    worldj::Vector{Int64}
    worldi::Vector{Int64}
    targetrank::Int64
    originrank::Int64
    values::Vector{Float64}
end


struct ActualMessage{N2}
    worldj::SVector{N2,Int64}
    worldi::SVector{N2,Int64}
    values::SVector{N2,Float64}
    len::Int64
    originrank::Int64
end

function preparemessage(gridm::GridMessage)::ActualMessage
    ele = length(gridm.worldj)
    ActualMessage{ele}(
        SVector{ele,Float64}( vcat(gridm.worldj)),
        SVector{ele,Float64}( vcat(gridm.worldi)),
        SVector{ele,Float64}( vcat(gridm.values)),
        ele,localLattice.sendMessages[2].originrank)
end



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
    # const x_0::Vector{T}
    # const v_0::Vector{T}
    grid::Matrix{T}
    phaseTemp::Matrix{T} = zeros(Float64,ΔNv,ΔNx)
    new_localji::Vector{Int64} = zeros(Int64,2)
    new_tempji::Vector{Int64} = zeros(Int64,2)
    new_globalji::Vector{Int64} = zeros(Int64,2)
    new_target::Int64 = -1
    pixel_value::T = zero(Float64)
    ρ::Vector{T} = integrate_lattice(grid,dv)
    a::Vector{T} = zeros(Float64,Nx)
    # sendip::Matrix{Int64}
    # sendjp::Matrix{Int64}
    # sendvaluesp::Matrix{Int64}
    sendMessages::Vector{GridMessage}
    recvMessages::Vector{GridMessage}
    # recvMessages::Vector{Vector{ActualMessage}}
end

function resetsendMessages(localLattice::ParallelLattice,simTopo::SimulationTopology)
    localLattice.sendMessages = [GridMessage(Int64[],Int64[],i,simTopo.topo.rank,Int64[]) for i in 0:simTopo.topo.nworkers-1 ]
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


"""
    calculate_new_pos!(i::Int64,j::Int64,sim::Lattice)::Bool

TBW
"""
function calculate_new_pos!(i::Int64,j::Int64,sim::Lattice)::Bool
    sim.new_ji[1] = j + Int(round(sim.a[i]*sim.dt/sim.dv))
    if !(oneunit(Int64) < sim.new_ji[1] < sim.Nv)
        return false
    end
    # new_i = i + Int(round(vel(new_j;V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))
    sim.new_ji[2] = mod(i + Int(round(vel(sim.new_ji[1];V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))-oneunit(i),sim.Nx)+oneunit(i)
    true
end

"""
    streamingStep!!(sim::Lattice)

TBW
"""
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
function parallelCalculate_new_pos!(j::Int64,i::Int64,localLattice::ParallelLattice, simTopo::SimulationTopology)::Bool
    # localLattice.new_globalji[2] = 0
    # localLattice.new_globalji[1] = 0
    localLattice.new_tempji[1] = j + Int(round(
        localLattice.a[locali2globali(i,simTopo)]*localLattice.dt/localLattice.dv))
    localLattice.new_globalji[1] = localj2globalj(localLattice.new_tempji[1],simTopo)
    if !(oneunit(Int64) < localLattice.new_globalji[1] < localLattice.Nv)
        return false
    end
    # new_i = i + Int(round(vel(new_j;V_min=sim.V_min,dv=sim.dv)*sim.dt/sim.dx))
    localLattice.new_globalji[2] = mod(i + Int(round(vel(
        localLattice.new_tempji[1];V_min=localLattice.V_min,dv=localLattice.dv)*localLattice.dt/localLattice.dx))-oneunit(i),localLattice.Nx)+oneunit(i)
    true
end


"""
    updatenewlocaljl!(localLattice::ParallelLattice,simTopo::SimulationTopology)

Loads the new_localji using new_globalji. Returns the rank of the process for which that pixel is local.
"""
function updatenewlocaljl!(localLattice::ParallelLattice,simTopo::SimulationTopology)::Int64
    rank, localLattice.new_localji = globalji2rankjivect(localLattice.new_globalji[1],localLattice.new_globalji[2],simTopo)
    rank
end 


"""
    updateormessage!(j::Int64,i::Int64,targetrank::Int64, llocalLattice::ParallelLattice,simTopo::SimulationTopology)

Updates the local grid or adds the pixel to the message corresponding to the target process.
"""
function updateormessage!(j::Int64,i::Int64,targetrank::Int64, llocalLattice::ParallelLattice,simTopo::SimulationTopology)
    # if targetrank == simTopo.topo.rank
    if false
        llocalLattice.phaseTemp[llocalLattice.new_localji[1],llocalLattice.new_localji[2]] += llocalLattice.grid[j,i]
    else
        push!(llocalLattice.sendMessages[targetrank+1].worldj,llocalLattice.new_globalji[1])
        push!(llocalLattice.sendMessages[targetrank+1].worldi,llocalLattice.new_globalji[2])
        push!(llocalLattice.sendMessages[targetrank+1].values,llocalLattice.pixel_value)
    end
    nothing
end

"""
    parallel_streamingStep!!(localLattice::ParallelLattice,simTopo::SimulationTopology)

Iterates over the local grid doing the streaming step. It update phaseTemp if the new pixel is in the same
node, otherwise prepares a message to the correct node.
"""
function parallel_streamingStep!!(localLattice::ParallelLattice,simTopo::SimulationTopology)
    # @show "im inside parallel_streamingStep!!"
    for i in 1:size(localLattice.grid,2)
        for j in 1:size(localLattice.grid,1)
            if !parallelCalculate_new_pos!(j,i,localLattice,simTopo)
                continue
            end
            updateormessage!(j,i,updatenewlocaljl!(localLattice,simTopo),localLattice,simTopo)
        end
    end
    nothing
end




"""
    sendGridUpdates(localLattice::ParallelLattice,simTopo::SimulationTopology, comm::MPI.Comm)

TBW
"""
function sendGridUpdates(localLattice::ParallelLattice,simTopo::SimulationTopology, comm::MPI.Comm)
    # sendMessages = Array{MPI.Request}(undef,simTopo.topo.nworkers)
    # recvMessages = Array{MPI.Request}(undef,simTopo.topo.nworkers)
    sendMessages = MPI.Request[]

    # @show "I will send from $(simTopo.topo.otherworkers)"
    for target in simTopo.topo.otherworkers
        push!(sendMessages, MPI.isend(localLattice.sendMessages[target+1], comm; dest=target, tag=1254))
    end
    sendMessages
end

"""
    imprint_streamingStep!(localLattice::ParallelLattice)

Should always be executed after an streamingStep. It moves the data from phaseTemp to grid.
"""
function imprint_streamingStep!(localLattice::ParallelLattice)
    for i in 1:size(localLattice.grid,2)
        for j in 1:size(localLattice.grid,1)
            localLattice.grid[j,i] = localLattice.phaseTemp[j,i]
            localLattice.phaseTemp[j,i] = zero(Float64)
        end
    end
end

"""
    addmessagestogrid(localLattice::ParallelLattice)

Add the contribution from the other processes into the local grid.
"""
function addmessagestogrid(localLattice::ParallelLattice)
    # @show "I will start updating the grid"
    for mes in localLattice.recvMessages
        for k in 1:length(mes.worldj)
           localLattice.grid[globalji2rankjinorank(mes.worldj[k],mes.worldi[k],simTopo)...] += mes.values[k]
        end
    end
end


"""
    receiveGridUpdates!(localLattice::ParallelLattice,simTopo::SimulationTopology,comm::MPI.Comm)

TBW
"""
function receiveGridUpdates!(localLattice::ParallelLattice,simTopo::SimulationTopology,comm::MPI.Comm)
    for sender in simTopo.topo.otherworkers
        localLattice.recvMessages[sender+1] = MPI.recv(comm;source=sender, tag=1254)::GridMessage
    end
end

"""
    receiveGridUpdatesBetter!(localLattice::ParallelLattice,simTopo::SimulationTopology,comm::MPI.Comm)

TBW
"""
function receiveGridUpdatesBetter!(localLattice::ParallelLattice,simTopo::SimulationTopology,comm::MPI.Comm)
    # (true, MPI.deserialize(buf), stat)
    received = 0
    for sender in simTopo.topo.otherworkers
        boolstatus,message,_ = irecv(sender,1254,comm)
        if boolstatus
            received+=1
            localLattice.recvMessages[sender+1] = message
        end
    end
end

"""
    multinodeStreamingStep!(localLattice::ParallelLattice,simTopo::SimulationTopology,comm::MPI.Comm)

Does a MPI parallel streaming step. 
"""
function multinodeStreamingStep!(localLattice::ParallelLattice,simTopo::SimulationTopology,comm::MPI.Comm)
    # @show "im inside multinodeStreamingStep"
    parallel_streamingStep!!(localLattice,simTopo)
    reqs = sendGridUpdates(localLattice,simTopo, comm)
    imprint_streamingStep!(localLattice)
    receiveGridUpdates!(localLattice,simTopo,comm)
    # receiveGridUpdatesBetter!(localLattice,simTopo,comm)
    addmessagestogrid(localLattice::ParallelLattice)
    resetsendMessages(localLattice,simTopo)
end


# function shouldIevensendamessage(localLattice::ParallelLattice)
#     localLattice
# end

"""
    parallelIntegrate_steps!(localLattice::ParallelLattice,sim::Union{Lattice,Nothing},simTopo::SimulationTopology)

Time evolves the simulation sim.Nt number of steps.
"""
function parallelIntegrate_steps!(localLattice::ParallelLattice,sim::Union{Lattice,Nothing},simTopo::SimulationTopology)
    for i in 1:localLattice.Nt
        integrate_lattice!(localLattice.ρ, localLattice.grid, localLattice.dv)
        syncronizedensity(localLattice, sim, simTopo,comm)
        if simTopo.topo.rank == 0
            sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*sim.G) #parallel version
            sim.a = -num_diff(sim.Φ,1,5,sim.dx) #parallel version
        end
        MPI.Bcast!(localLattice.a, 0, comm)
        multinodeStreamingStep!(localLattice,simTopo,comm)
    end
    nothing
end

function parallelSimulate!(localLattice::ParallelLattice,sim::Union{Lattice,Nothing},simTopo::SimulationTopology; t0::Float64 = 0.0)
    parallelIntegrate_steps!(localLattice,sim,simTopo)
    localLattice.Nt * localLattice.dt + t0
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

