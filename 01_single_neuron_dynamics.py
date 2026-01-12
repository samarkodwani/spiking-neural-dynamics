"""
Hodgkin-Huxley Neuron Simulation (1952)
---------------------------------------
Implementation of the conductance-based model using Euler integration.
Simulates a 'Double Pulse' protocol to demonstrate the absolute refractory period.
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Model Parameters
# ==========================================

# Membrane Capacitance
C_m = 1.0  # uF/cm^2

# Maximum Conductances
g_Na_max = 120.0  # mS/cm^2
g_K_max  = 36.0   # mS/cm^2
g_L      = 0.3    # mS/cm^2

# Nernst Potentials
E_Na = 50.0       # mV
E_K  = -77.0      # mV
E_L  = -54.387    # mV

# ==========================================
# 2. Simulation Setup
# ==========================================

time_duration = 50.0   # ms
dt = 0.01              # ms (Time step for stability)
t = np.arange(0, time_duration + dt, dt)

# State Variables
V = np.zeros(len(t))
n = np.zeros(len(t))   # K activation
m = np.zeros(len(t))   # Na activation
h = np.zeros(len(t))   # Na inactivation

# ==========================================
# 3. Gating Kinetics (Alpha/Beta functions)
# ==========================================

def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80)

def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V):  return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V):  return 1 / (1 + np.exp(-(V + 35) / 10))

# ==========================================
# 4. Initialization (Steady State @ -65mV)
# ==========================================

V[0] = -65.0
n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))

print("System Initialized. Starting Simulation...")

# ==========================================
# 5. Main Loop (Euler Method)
# ==========================================

for i in range(1, len(t)):

    # --- Stimulus Protocol ---
    # Pulse 1: 10-11ms (Elicit spike)
    # Pulse 2: 15-16ms (Test for Refractory Period failure)
    
    if 10.0 <= t[i-1] <= 11.0:
        I_inj = 10.0  # uA/cm^2
    elif 15.0 <= t[i-1] <= 16.0:
        I_inj = 10.0
    else:
        I_inj = 0.0

    # --- Update Gating Variables ---
    
    # Potassium (n)
    dn = alpha_n(V[i-1]) * (1 - n[i-1]) - beta_n(V[i-1]) * n[i-1]
    n[i] = n[i-1] + dn * dt

    # Sodium (m, h)
    dm = alpha_m(V[i-1]) * (1 - m[i-1]) - beta_m(V[i-1]) * m[i-1]
    m[i] = m[i-1] + dm * dt

    dh = alpha_h(V[i-1]) * (1 - h[i-1]) - beta_h(V[i-1]) * h[i-1]
    h[i] = h[i-1] + dh * dt

    # --- Update Voltage ---
    
    g_Na = g_Na_max * (m[i-1]**3) * h[i-1]
    g_K  = g_K_max  * (n[i-1]**4)
    
    I_Na = g_Na * (V[i-1] - E_Na)
    I_K  = g_K  * (V[i-1] - E_K)
    I_L  = g_L  * (V[i-1] - E_L)

    dV = (I_inj - I_Na - I_K - I_L) / C_m
    V[i] = V[i-1] + dV * dt

print("Simulation Complete.")

# ==========================================
# 6. Visualization
# ==========================================

plt.figure(figsize=(10, 8))

# Voltage Trace
plt.subplot(2, 1, 1)
plt.plot(t, V, 'k', linewidth=1.5)
plt.title('Hodgkin-Huxley Dynamics: Refractory Period Test')
plt.ylabel('Voltage (mV)')
plt.grid(True, alpha=0.3)

# Highlight Stimuli Regions
plt.axvspan(10, 11, color='r', alpha=0.1, label='Stimulus 1')
plt.axvspan(15, 16, color='k', alpha=0.1, label='Stimulus 2')
plt.legend(loc='upper right')

# Gating Variables
plt.subplot(2, 1, 2)
plt.plot(t, m, 'r', label='m (Na activation)')
plt.plot(t, h, 'g', label='h (Na inactivation)')
plt.plot(t, n, 'b', label='n (K activation)')
plt.title('Gating Variables')
plt.ylabel('Open Probability')
plt.xlabel('Time (ms)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
