

anim = @animate for i in 1:sim.Nt
    heatmap(sim.grid)
    sim.ρ = integrate_lattice!(zeros(size(sim.grid)[2]), sim.grid,sim.dv)
    sim.Φ = solve_f(sim.ρ .- mean(sim.ρ), sim.L, 4*π*G)
    sim.a = -num_diff(sim.Φ,1,5,sim.dx)
    rotate_pos!(sim.grid, Vector(v_0))
    rotate_vel!(sim.grid, (-sim.Nv .+ Int32.((round.(sim.a/sim.dv * sim.dt)))) .% sim.Nv)
end every 1

gif(anim, "/tmp/anim_fps15.gif", fps = 1)
