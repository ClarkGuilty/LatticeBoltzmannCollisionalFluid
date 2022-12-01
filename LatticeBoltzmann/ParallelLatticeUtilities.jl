function send2root(arr::Vector{Float64},tag::Int64,comm::MPI.Comm)
    MPI.Isend(arr, comm; dest=0, tag)
end

function recvfromall(tag::Int64,simTopo::SimulationTopology,comm::MPI.Comm)
    sreqs_workers = Array{MPI.Request}(undef,simTopo.topo.nworkers-1)
    buffers = Vector{Vector{Float64}}(undef,simTopo.topo.nworkers-1)
    for rank in 1:simTopo.topo.nworkers-1
        # @show rank
        buffers[rank] = zeros(Float64,simTopo.graphofdims[rank+1][2])
        sreqs_workers[rank] = MPI.Irecv!(buffers[rank], comm; source=rank, tag=tag)
    end
    # @show sreqs_workers
    buffers, sreqs_workers
end

"""
    syncronizedensity(localLattice::ParallelLattice, sim::Union{Lattice,Nothing}, simTopo::SimulationTopology,comm::MPI.Comm)

Synchronizes all the fragments of the density into the root process.
"""
function syncronizedensity(localLattice::ParallelLattice, sim::Union{Lattice,Nothing}, simTopo::SimulationTopology,comm::MPI.Comm)
    if simTopo.topo.rank != 0
        req = send2root(localLattice.ρ,DENSITYTAG,comm)
        MPI.Waitall([req])
    end
    ##
    if simTopo.topo.rank == 0
        buffers, sreqs_workers = recvfromall(DENSITYTAG,simTopo,comm)
        sim.ρ .= 0
        view(sim.ρ,tuple2range(rangeji(0,simTopo)[2])) .+= localLattice.ρ
        stats = MPI.Waitall(sreqs_workers) #TODO: try using MPI.WaitAny and meanwhile update sim.ρ.
        
        for (rank,buffer) in enumerate(buffers)
            view(sim.ρ,tuple2range(rangeji(rank,simTopo)[2])) .+= buffer
        end
    
        # @show sum(sim.ρ.*dx)
    end
    nothing
end


