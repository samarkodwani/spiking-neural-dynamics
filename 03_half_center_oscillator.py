import numpy as np
import matplotlib.pyplot as plt

# --- Class Definition---
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

# SETUP: Two identical neurons inhibited by each other
# RS Parameters provide the adaptation ('d') needed for fatigue
n1 = IzhiNeuron(a=0.02, b=0.2, c=-65, d=8)
n2 = IzhiNeuron(a=0.02, b=0.2, c=-65, d=8)

# SYMMETRY BREAKING
# We force one neuron to be closer to threshold so it "wins" the first cycle
n1.v = -60.0 
n2.v = -70.0

# OSCILLATOR PARAMETERS
w = 20.0        # Strong Mutual Inhibition (The "Punch")
tau_g = 10.0    # Slower decay allows longer burst durations
E_syn = -80.0   # GABAergic Reversal Potential
I_drive = 5.0   # Constant DC drive to both neurons

g12 = 0.0 # Synapse N1 -> N2
g21 = 0.0 # Synapse N2 -> N1

T = 1000; dt = 0.1
time = np.arange(0, T, dt)
v1_hist, v2_hist = [], []

for t in time:
    # 1. DECAY CONDUCTANCES
    g12 *= np.exp(-dt / tau_g)
    g21 *= np.exp(-dt / tau_g)
     
    # 2. CALCULATE CROSS-INHIBITION
    # N1 is inhibited by N2 (g21)
    # N2 is inhibited by N1 (g12)
    # The 'Driving Force' (v - E_syn) ensures current is zero if neuron is already at -80mV
    I_to_n1 = -g21 * (n1.v - E_syn)
    I_to_n2 = -g12 * (n2.v - E_syn)
     
    # 3. UPDATE NEURONS
    # Input = Constant Drive + Synaptic Inhibition
    spike1 = n1.step(dt, I_drive + I_to_n1)
    spike2 = n2.step(dt, I_drive + I_to_n2)
     
    # 4. TRANSMISSION
    if spike1: g12 += w
    if spike2: g21 += w
         
    v1_hist.append(n1.v)
    v2_hist.append(n2.v)

# --- PLOTTING (Anti-Phase Synchronization) ---
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(time, v1_hist, 'b', label='Neuron 1')
plt.title('Half-Center Oscillator (Central Pattern Generator)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(time, v2_hist, 'r', label='Neuron 2')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.grid(True)
plt.show()