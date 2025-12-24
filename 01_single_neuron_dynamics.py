import numpy as np
import matplotlib.pyplot as plt

# NEURON PARAMETERS (Izhikevich 2003)
# RS = Regular Spiking (Cortex Pyramidal): Adapts over time
# FS = Fast Spiking (Interneuron): No adaptation, high frequency
# CH = Chattering (Visual Cortex): High freq bursts
neuron_types = {
    'RS': {'a': 0.02, 'b': 0.2,  'c': -65, 'd': 8},
    'FS': {'a': 0.1,  'b': 0.2,  'c': -65, 'd': 2},
    'CH': {'a': 0.02, 'b': 0.2,  'c': -50, 'd': 2}
}

class IzhiNeuron:
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v = -65.0      # Membrane Potential (mV)
        self.u = b * self.v # Recovery Variable (pA) - Represents K+ channel activation
        self.spike_count = 0

    def step(self, dt, I):
        # 1. SOLVE DIFFERENTIAL EQUATIONS (Forward Euler Method)
        
        # Voltage Equation: dv/dt = 0.04v^2 + 5v + 140 - u + I
        # The quadratic term (v^2) creates the "tipping point" for the spike (threshold behavior)
        dv_dt = (0.04 * self.v**2) + (5 * self.v) + 140 - self.u + I
        
        # Recovery Equation: du/dt = a(bv - u)
        # 'a' determines how fast the neuron recovers (time scale)
        # 'b' determines the sensitivity of recovery to sub-threshold voltage
        du_dt = self.a * (self.b * self.v - self.u)

        self.v += dv_dt * dt
        self.u += du_dt * dt

        # 2. SPIKE & RESET DYNAMICS
        # If voltage hits +30mV, we assume a spike occurred (Na+ channels snap open)
        if self.v >= 30:
            self.v = self.c      # Reset membrane potential (After-Hyperpolarization)
            self.u += self.d     # Step increase in recovery (Simulates fatigue/adaptation)
            self.spike_count += 1
            return True
            
        return False

# --- SIMULATION ---
current_type = 'RS' # Try changing to 'FS' or 'CH' to see different firing patterns
params = neuron_types[current_type]
neuron = IzhiNeuron(**params)

T = 1000; dt = 0.1
time = np.arange(0, T, dt)

# Step Input: 0 current until 100ms, then 10 pA DC current
I_input = np.zeros(len(time))
I_input[int(100/dt):] = 10 

v_hist, u_hist = [], []

for i, t in enumerate(time):
    neuron.step(dt, I_input[i])
    v_hist.append(neuron.v)
    u_hist.append(neuron.u)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(time, v_hist)
plt.title(f'{current_type} Neuron Dynamics')
plt.ylabel('Voltage (mV)')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(time, u_hist, color='orange')
plt.title('Recovery Variable (u) - Tracks Adaptation')
plt.xlabel('Time (ms)')
plt.ylabel('Recovery (pA)')
plt.grid(True)
plt.tight_layout()
plt.show()