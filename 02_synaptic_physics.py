import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETUP ---
class IzhiNeuron:
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v = -65.0
        self.u = self.b * self.v
        
    def step(self, dt, I):
        dv_dt = (0.04 * self.v**2) + (5 * self.v) + 140 - self.u + I
        du_dt = self.a * (self.b * self.v - self.u)
        self.v += dv_dt * dt
        self.u += du_dt * dt
        if self.v >= 30:
            self.v = self.c
            self.u += self.d
            return True
        return False

# Create Two Neurons
# N1: The Sender (Chattering Type to show multiple inputs)
n1 = IzhiNeuron(a=0.02, b=0.2, c=-50, d=2) 
# N2: The Receiver (Regular Spiking)
n2 = IzhiNeuron(a=0.02, b=0.2, c=-65, d=8)

# Synapse Parameters
w = 1.0         # Synaptic Weight (Strength)
tau_g = 10.0    # Decay constant (ms) - How fast channels close
E_syn = 0.0     # Reversal Potential: 0mV = Excitatory (AMPA)
g = 0.0         # Conductance starts at 0

# Time
T = 200         # Short duration to zoom in on the physics
dt = 0.1
time = np.arange(0, T, dt)

# Input: Short kick to N1 to make it fire
I_input = np.zeros(len(time))
I_input[int(20/dt):int(40/dt)] = 20 

# Storage
v1_hist = []
v2_hist = []
g_hist  = []

# --- 2. MAIN LOOP ---
for i, t in enumerate(time):
    # A. Synaptic Physics (Kinetic Model)
    # The conductance 'g' decays exponentially over time (channels closing)
    dg_dt = -g / tau_g
    g += dg_dt * dt
    
    # B. Calculate Current (Ohm's Law)
    # The current entering N2 depends on the conductance 'g' and the driving force (V - E)
    I_syn = -g * (n2.v - E_syn)
    
    # C. Update Neurons
    spike1 = n1.step(dt, I_input[i]) # N1 gets external input
    n2.step(dt, I_syn)               # N2 gets synaptic input
    
    # D. Transmission
    # If N1 spikes, neurotransmitters are released -> g increases
    if spike1:
        g += w

    # Store Data
    v1_hist.append(n1.v)
    v2_hist.append(n2.v)
    g_hist.append(g)

# --- 3. PLOTTING ---
plt.figure(figsize=(10, 8))

# Plot 1: The Sender
plt.subplot(3, 1, 1)
plt.plot(time, v1_hist, 'b')
plt.title('Step 1: Pre-Synaptic Neuron Fires (The Trigger)') 
plt.ylabel('Voltage (mV)')
plt.grid(True)

# Plot 2: The Mechanism (Most Important for Biophysics)
plt.subplot(3, 1, 2)
plt.plot(time, g_hist, 'g')
plt.title('Step 2: Synaptic Conductance (The Mechanism)')
plt.ylabel('Conductance (S)')
plt.text(50, max(g_hist)*0.5, "Exponential Decay (Channel Closing)", color='green', fontsize=10)
plt.grid(True)

# Plot 3: The Receiver
plt.subplot(3, 1, 3)
plt.plot(time, v2_hist, 'r')
plt.title('Step 3: Post-Synaptic Response') 
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.grid(True)

plt.tight_layout()
plt.show()