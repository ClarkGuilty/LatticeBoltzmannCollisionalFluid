include("LatticeBoltzmann.jl")
##
N = 1024
Nx = N
Nv = N
Nt = 25
v_min = -1.0
v_max = 1.0
x_min = -0.5
x_max = 0.5
lv = v_max - v_min
lx = x_max - x_min
dv = lv / (Nv)
dx = lx / (Nx)
dt = 0.1 * dx/dv
# dt = 0.2
G = 1.0
v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
x_0 = Float64.(LinRange(x_min,x_max,Nx+1)[1:end-1])
##
# v_0[end]
# v_min + dv*2047
# vel(2047;V_min=v_min,dv=dv)

# heatmap(grid,aspect_ratio=:equal)
# density = zeros(typeof(grid[end,end]), size(grid,2))
# @btime integrate_lattice!(density,grid,dv)
# integrate_lattice!(density,grid,dv)
# plot(density)

##
sim = Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = 5,
                dt = 0.1, V_min=v_min, V_max=v_max, G = 0.05,
                #grid = jeans.(x_0', v_0, σ=σ,ρ=ρ,k=k,A=A))
                grid = gaussian_2d.(x_0',v_0))
#
heatmap(sim.grid,title="Base",aspect_ratio=:equal)
plot!([sim.Nx/2,sim.Nx/2],[0,sim.Nv],color=:green)
plot!([0,sim.Nx],[sim.Nv/2,sim.Nv/2],color=:green)
##
# v_0 = Float64.(LinRange(v_min,v_max,Nv+1)[1:end-1])
# v_1 = vel.(1:Nv;V_min=sim.V_min,dv=sim.dv) 
# v_0 .- v_1
simulate!(sim)
#
heatmap(sim.grid,title="Base",aspect_ratio=:equal)
plot!([sim.Nx/2,sim.Nx/2],[0,sim.Nv],color=:green)
plot!([0,sim.Nx],[sim.Nv/2,sim.Nv/2],color=:green)
##
x_0[argmax(sim.ρ)]
VSCodeServer.@profview simulate!(sim)
@btime simulate!(sim)
heatmap(sim.grid,aspect_ratio=:equal,origin=:upper)

##
# sim = Lattice(X_min = x_min, X_max = x_max, Nx = Nx, Nv = Nv, Nt = 1,
#                 dt = 0.1, V_min=v_min, V_max=v_max, G = 0.05,
#                 #grid = jeans.(x_0', v_0, σ=σ,ρ=ρ,k=k,A=A))
#                 grid = gaussian_2d.(x_0',v_0))
sim = Lattice(X_min = x_min, X_max = x_max, Nx = N, Nv = N, Nt = 1,
                dt = 0.1, V_min=v_min, V_max=v_max, G = 0.05,
                grid = bullet_cluster.(x_0',v_0;x0=-0.2,x1=0.2,σv1=0.08,σv2=0.08,σx1=0.08,σx2=0.08,A1=10,A2=10))

@time anim = @animate for i in 1:200
    heatmap(sim.grid)
    simulate!(sim)
end every 2
gif(anim, "/tmp/anim_fps15.gif", fps = 10)