import numpy as np
import matplotlib.pyplot as plt

# Parameters
N0 = 10000 # change this to 1000 if PC is old
num_simulations = 100 # reduce this to 10 if PC is old
half_life = 0.8387
max_time = 1
dt = 0.001

# Read more about metropolis decay algorithm here - https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/

def metropolis_decay(N0, half_life, max_time, dt):
    decay_constant = np.log(2) / half_life
    time_points = np.arange(0, max_time, dt)
    num_steps = len(time_points)
    decayed_particles = np.zeros(num_steps)
    undecayed_particles = np.ones(num_steps) * N0

    for i in range(1, num_steps):
        undecayed_previous = int(undecayed_particles[i - 1])
        decayed_previous = int(decayed_particles[i - 1])

        decay_probability = 1 - np.exp(-decay_constant * dt)

        random_numbers = np.random.rand(undecayed_previous)
        decayed_this_step = random_numbers < decay_probability # Boolean array for decayed particles

        undecayed_particles[i] = undecayed_previous - np.sum(decayed_this_step)
        decayed_particles[i] = decayed_previous + np.sum(decayed_this_step)

    return time_points, undecayed_particles, decayed_particles


time_points, undecayed_particles, decayed_particles = metropolis_decay(N0, half_life, max_time, dt)

# Plot the results
plt.plot(time_points, undecayed_particles, label='Undecayed Particles')
plt.plot(time_points, decayed_particles, label='Decayed Particles')
plt.xlabel('Time (s)')
plt.ylabel('Number of Particles')
plt.title('Radioactive Decay of $^8$Li')
plt.xlim(0, max_time)
plt.ylim(0, N0)
plt.grid()
plt.legend()
plt.savefig("LiDecay.png")
