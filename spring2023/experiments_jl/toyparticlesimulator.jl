using Random
using Distances
using Statistics
using LinearAlgebra
using PyPlot
using PyCall
import PyCall.pyimport
mpl = pyimport("matplotlib")
mpl[:use]("Agg")
#include("real_data_testing.jl")


const X, Y = 1, 2

mutable struct MDSimulation
    pos::Matrix{Float64}
    vel::Matrix{Float64}
    n::Int
    r::Float64
    m::Float64
    nsteps::Int
end


function advance(sim::MDSimulation, dt)
    sim.nsteps += 1
    sim.pos .+= sim.vel .* dt
    dist = zeros(sim.n, sim.n)
    iarr = []
    jarr = []
    for i in 1:sim.n
        for j in 1:i-1
            dist[i,j] = norm(sim.pos[i] - sim.pos[j], 2)
            dist[j,i] = norm(sim.pos[i] - sim.pos[j], 2)
            if dist[i, j] < 2 * sim.r 
                push!(iarr, i)
                push!(jarr, j)
            end
        end 
    end

    for (i, j) in zip(iarr, jarr)
        pos_i, vel_i = sim.pos[i, :], sim.vel[i, :]
        pos_j, vel_j = sim.pos[j, :], sim.vel[j, :]
        rel_pos, rel_vel = pos_i - pos_j, vel_i - vel_j
        r_rel = dot(rel_pos, rel_pos)
        v_rel = dot(rel_vel, rel_pos)
        v_rel = 2 * rel_pos * v_rel / r_rel - rel_vel
        v_cm = (vel_i + vel_j) / 2
        sim.vel[i, :] = v_cm - v_rel/2
        sim.vel[j, :] = v_cm + v_rel/2
    end

    hit_left_wall = sim.pos[:, X] .< sim.r
    hit_right_wall = sim.pos[:, X] .> 1 - sim.r
    hit_bottom_wall = sim.pos[:, Y] .< sim.r
    hit_top_wall = sim.pos[:, Y] .> 1 - sim.r
    sim.vel[hit_left_wall .| hit_right_wall, X] .*= -1
    sim.vel[hit_bottom_wall .| hit_top_wall, Y] .*= -1
end

n = 1000
rscale = 5e6
r = 2e-10 * rscale
tscale = 1e9
sbar = 353 * rscale / tscale
FPS = 30
dt = 1 / FPS
m = 1

pos = rand(n, 2)
theta = rand(n) * 2 * π
s0 = sbar * rand(n)
vel = hcat(s0 .* cos.(theta), s0 .* sin.(theta))

sim = MDSimulation(pos, vel, n, r, m, 0)


#the code below is meant for visualizing the simulator
pygui(true)
DPI = 100
width, height = 1000, 500
fig = figure(figsize=(width/DPI, height/DPI), dpi=DPI)
fig.subplots_adjust(left=0, right=0.97)
sim_ax = fig.add_subplot(121, aspect="equal", autoscale_on=false)
sim_ax.set_xticks([])
sim_ax.set_yticks([])


speed_ax = fig.add_subplot(122)
speed_ax.set_xlabel(L"Speed $v\,/m\,s^{-1}$")
speed_ax.set_ylabel("\$f(v)\$")
particles, = sim_ax.plot([], [], "ko")

# Create a function to update the animation
function update_animation(i)
    global sim
    advance(sim, dt)
    particles.set_data(sim.pos[:, X], sim.pos[:, Y])
    particles.set_markersize(0.5)
    return particles
end

# Create an animation
frames = 1000
pyimport("matplotlib.animation")  # Import the animation module
anim = matplotlib.animation.FuncAnimation(fig, update_animation, frames=frames, interval=0, blit=false)


# Show animation
PyPlot.show()
velocities = sim.vel

