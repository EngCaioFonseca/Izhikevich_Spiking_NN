import numpy as np
import matplotlib.pyplot as plt

class IzhikevichNeuron:
    def __init__(self, a, b, c, d):
        self.a = a  # Time scale of the recovery variable
        self.b = b  # Sensitivity of the recovery variable
        self.c = c  # After-spike reset value of membrane potential
        self.d = d  # After-spike reset of the recovery variable
        self.v = self.c  # Initial membrane potential
        self.u = self.b * self.v  # Initial recovery variable
        self.spike_times = []

    def update(self, I, t, dt):
        if self.v >= 30:  # Spike threshold
            self.spike_times.append(t)
            self.v = self.c
            self.u += self.d
        
        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + I
        du = self.a * (self.b * self.v - self.u)
        
        self.v += dv * dt
        self.u += du * dt
        
        # Ensure v stays within a reasonable range
        self.v = np.clip(self.v, -100, 30)
        
        return self.v, self.u

def simulate_neuron(neuron, I, simulation_time, dt):
    n = int(simulation_time / dt)
    v = np.zeros(n)
    u = np.zeros(n)
    
    for i in range(n):
        v[i], u[i] = neuron.update(I, i*dt, dt)
    
    return v, u, neuron.spike_times

# Simulate a single Izhikevich neuron
neuron = IzhikevichNeuron(0.02, 0.2, -65, 8)  # Regular spiking neuron
I = 10  # Input current
simulation_time = 1000  # ms
dt = 0.1  # ms

v, u, spike_times = simulate_neuron(neuron, I, simulation_time, dt)

# Plot the results
t = np.arange(0, simulation_time, dt)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, v)
plt.title('Membrane Potential')
plt.ylabel('Voltage (mV)')
plt.subplot(2, 1, 2)
plt.plot(t, u)
plt.title('Recovery Variable')
plt.xlabel('Time (ms)')
plt.ylabel('u')
plt.tight_layout()
plt.show()

# Spiking Neural Network
class SpikingNeuralNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [IzhikevichNeuron(0.02, 0.2, -65, 8) for _ in range(num_neurons)]
        self.weights = np.random.randn(num_neurons, num_neurons) * 0.1
        
    def simulate(self, external_current, simulation_time, dt):
        n = int(simulation_time / dt)
        v_history = np.zeros((self.num_neurons, n))
        u_history = np.zeros((self.num_neurons, n))
        spike_history = np.zeros((self.num_neurons, n), dtype=bool)
        
        for i in range(n):
            for j in range(self.num_neurons):
                I = external_current[j] + np.sum(self.weights[j] * spike_history[:, i-1])
                v, u = self.neurons[j].update(I, i*dt, dt)
                v_history[j, i] = v
                u_history[j, i] = u
                spike_history[j, i] = v >= 30
        
        return v_history, u_history, spike_history

# Simulate the Spiking Neural Network
num_neurons = 5
snn = SpikingNeuralNetwork(num_neurons)
external_current = np.array([10, 8, 12, 5, 15])
simulation_time = 1000  # ms
dt = 0.1  # ms

v_history, u_history, spike_history = snn.simulate(external_current, simulation_time, dt)

# Plot the results for selected neurons
t = np.arange(0, simulation_time, dt)
selected_neurons = [0, 2, 4]  # Select neurons to display

plt.figure(figsize=(15, 10))
for i, neuron_id in enumerate(selected_neurons):
    plt.subplot(len(selected_neurons), 1, i+1)
    plt.plot(t, v_history[neuron_id])
    plt.title(f'Neuron {neuron_id} Membrane Potential')
    plt.ylabel('Voltage (mV)')
    if i == len(selected_neurons) - 1:
        plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show()

# Plot raster plot of spikes
plt.figure(figsize=(12, 6))
for i in range(num_neurons):
    spike_times = np.where(spike_history[i])[0] * dt
    plt.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=10)
plt.title('Spike Raster Plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')
plt.ylim(-0.5, num_neurons - 0.5)
plt.show()
