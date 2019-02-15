import os
import numpy as np
import readdy
import myintegrator as mi

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt


n_particles = 500
out_file = "anisotropic_msd.h5"
origin = np.array([-40., -40., -40.])
extent = np.array([80., 80., 80.])
timestep = 0.01

# Create the custom integrator, with desired arguments, here alpha_x=0.3, alpha_y=0.6, alpha_z=1
custom_integrator = mi.SCPUAnisotropicBD(timestep, 0.3, 0.6, 1.)
system = readdy.ReactionDiffusionSystem(box_size=extent + 10., unit_system=None)
# the scalar diffusion constant is 1 here and will be multiplied with the
# alpha values component-wise as defined in `binding.cpp`
system.add_species("A", diffusion_constant=1.)
system.potentials.add_box("A", 50., origin, extent)

simulation = system.simulation(kernel="SingleCPU")

# Inject the custom integrator
simulation.integrator = custom_integrator

simulation.observe.particle_positions(1)
init_pos = np.random.normal(size=(n_particles, 3))
simulation.add_particles("A", init_pos)

simulation.output_file = out_file
if os.path.exists(simulation.output_file):
    os.remove(simulation.output_file)
simulation.run(1000, timestep)

# Verify the results, calculate MSD component-wise
traj = readdy.Trajectory(out_file)
times, positions = traj.read_observable_particle_positions()

# convert positions to numpy array
T = len(positions)
N = len(positions[0])
pos = np.zeros(shape=(T, N, 3))
for t in range(T):
    for n in range(N):
        pos[t, n, 0] = positions[t][n][0]
        pos[t, n, 1] = positions[t][n][1]
        pos[t, n, 2] = positions[t][n][2]

difference = pos - init_pos
squared_displacements = difference * difference  # do not sum over coordinates
squared_displacements = squared_displacements.transpose()  # T x N x C-> C x N x T

mean = np.mean(squared_displacements, axis=1)

std_dev = np.std(squared_displacements, axis=1)
std_err = np.std(squared_displacements, axis=1) / np.sqrt(squared_displacements.shape[1])

stride = 20
plt.figure(figsize=(5, 4))
plt.errorbar((times * timestep)[::stride], mean[0][::stride], yerr=std_err[0][::stride], fmt=".")
plt.errorbar((times * timestep)[::stride], mean[1][::stride], yerr=std_err[1][::stride], fmt=".")
plt.errorbar((times * timestep)[::stride], mean[2][::stride], yerr=std_err[2][::stride], fmt=".")
plt.plot(times * timestep, 2. * 0.3 * times * timestep, color="C0", label=r"$(x(t)-x_0)^2=2\times 0.3 \times t$")
plt.plot(times * timestep, 2. * 0.6 * times * timestep, color="C1", label=r"$(y(t)-y_0)^2=2\times 0.6 \times t$")
plt.plot(times * timestep, 2. * times * timestep, color="C2", label=r"$(z(t)-z_0)^2=2\times t$")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Mean squared displacement")
plt.title("Anisotropic diffusion - $(D_x,D_y,D_z)=(0.3,0.6,1)$")
plt.gcf().tight_layout()
plt.savefig("anisotropic-diffusion.png", dpi=150)
