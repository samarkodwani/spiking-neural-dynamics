"""
Synaptic Transmission Demo
--------------------------
Simulates a single unidirectional excitatory synapse (AMPA-like) between 
a chattering presynaptic neuron and a regular spiking postsynaptic neuron.
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Izhikevich Neuron Class
# ==========================================
class IzhiNeuron:
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v = -65.0
        self.u = self.b * self.v
        
    def step(self, dt, I):
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (self.a * (self.b * self.v - self.u)) * dt
        self.v += dv
        self.u += du
        
        if self.v >= 30:
            self.v = self.c
            self.u += self.d
            return True
        return False

# ==========================================
# 2. Network Setup
# ==========================================

# N1: Sender (Chattering)
n1 = IzhiNeuron(a=0.02, b=0.2, c=-50, d=2) 
# N2: Receiver (RS)
n2 = IzhiNeuron(a=0.02, b=0.2, c=-65, d=8)

# Synapse Parameters (Excitatory)
w = 1.0        # Synaptic Weight
tau_g = 10.0   # Conductance decay (ms)
E_syn = 0.0    # Reversal Potential (0mV = Excitatory)
g = 0.0        # Initial Conductance

# Simulation Parameters
T = 200
dt = 0.1
time = np.arange(0, T, dt)

# Input Protocol: 20ms pulse to trigger N1
I_input = np.zeros(len(time))
I_input[int(20/dt):int(40/dt)] = 20 

# Data Storage
v1_hist, v2_hist, g_hist = [], [], []

# ==========================================
# 3. Simulation Loop
# ==========================================
for i, t in enumerate(time):
    # Synaptic Dynamics (Exponential Decay)
    g += (-g / tau_g) * dt
    
    # Calculate Synaptic Current
    I_syn = -g * (n2.v - E_syn)
    
    # Update Neurons
    spike1 = n1.step(dt, I_input[i])
    n2.step(dt, I_syn)
    
    # Transmission Event
    if spike1:
        g += w

    v1_hist.append(n1.v)
    v2_hist.append(n2.v)
    g_hist.append(g)

# ==========================================
# 4. Visualization
# ==========================================
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, v1_hist, 'b')
plt.title('Presynaptic Neuron (N1)')
plt.ylabel('Voltage (mV)')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(time, g_hist, 'g')
plt.title('Synaptic Conductance (g)')
plt.ylabel('Siemens')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(time, v2_hist, 'r')
plt.title('Postsynaptic Response (N2)') 
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
