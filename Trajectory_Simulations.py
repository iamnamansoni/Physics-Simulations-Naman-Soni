#This work was done as part of General Relativity course presentation in which we discussed Photon Orbiting around Schwarzschild and Kerr Blackholes.
#The credit for this code goes to my group member Shivam Kumar (will mention his id here) and a little to me for suggestions. 
#We were able to run this properly on VS code so it is recommended.
#I have also uploaded the pdfs of theory and presentation in this repository.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define the constants
M = 5.0
R_S = 2 * M  # Schwarzschild Radius (Event Horizon)
R_PS = 3 * M  # Photon Sphere Radius

# Critical impact parameter
b_crit = np.sqrt(27) * M

# Photon trajectories parameters
params = {
    'critical': {'L': b_crit, 'E': 1.0, 'M': M, 'label': f'Orbiting Photons (b = b_crit = sqrt(27)*M)'},
    'scatter_1': {'L': b_crit + 0.05, 'E': 1.0, 'M': M, 'label': 'Escaped Photons (b > b_crit)'},
    'scatter_2': {'L': b_crit + 0.2, 'E': 1.0, 'M': M, 'label': ''},
    'capture_1': {'L': b_crit - 1.0, 'E': 1.0, 'M': M, 'label': 'Captured Photons (b < b_crit)'}
}

# Escaped Photons
for i in range(-16, -1, 2):
    key = f'scatter_{i}'
    params[key] = {
        'L': b_crit - float(i),
        'E': 1.0,
        'M': M,
        'label': f''
    }

# Captured Photons
for i in range(3, 11, 2):
    key = f'capture_{i}'
    params[key] = {
        'L': b_crit - float(i),
        'E': 1.0,
        'M': M,
        'label': f''
    }
    

def photon_orbit(lambda_param, Y, p):
    r, r_dot, phi = Y
    L, E, M = p['L'], p['E'], p['M']

    dr_dlambda = r_dot
    dr_dot_dlambda = (L**2 / r**3) - (3 * M * L**2 / r**4)
    dphi_dlambda = L / r**2

    return [dr_dlambda, dr_dot_dlambda, dphi_dlambda]

def horizon_event(t, Y, p):
    return Y[0] - (R_S + 1e-3)
horizon_event.terminal = True
horizon_event.direction = -1

# Initial conditions
r_0_initial = 50.0
t_span = [0, 600]

# Compute trajectories
trajectories = {}

# Colors dictionary with brighter colors
colors = {'scatter': 'blue', 'capture': 'red', 'critical': 'black'}
color1 = {'scatter': 'blue', 'capture': 'red', 'critical': 'black'}

for case, p in params.items():
    L = abs(p['L'])
    p['L'] = L

    if L / r_0_initial > 1:
        phi_0 = np.pi / 2
    else:
        phi_0 = np.arcsin(L / r_0_initial)

    schwarzschild_factor = (1 - 2 * p['M'] / r_0_initial)
    v_eff = schwarzschild_factor * (L**2 / r_0_initial**2)
    r_dot_0 = -np.sqrt(p['E']**2 - v_eff)

    Y0 = [r_0_initial, r_dot_0, phi_0]

    sol = solve_ivp(photon_orbit, t_span, Y0, args=(p,),
                    method='RK45', dense_output=True, rtol=1e-8, atol=1e-8, events=[horizon_event])

    # Reduced points for faster interpolation
    lambda_vals = np.linspace(t_span[0], sol.t[-1], 500)
    Y = sol.sol(lambda_vals)
    r = Y[0]
    phi = Y[2]

    valid_indices = r > R_S
    r_plot = r[valid_indices]
    phi_plot = phi[valid_indices]
    
    # Calculate arc length for uniform speed
    dr = np.diff(r_plot)
    dphi = np.diff(phi_plot)
    ds = np.sqrt(dr**2 + r_plot[:-1]**2 * dphi**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])

    trajectories[case] = {
        'r': r_plot,
        'phi': phi_plot,
        'arc_length': arc_length,
        'max_arc_length': arc_length[-1],
        'color': color1[case.split('_')[0]],
        'label': p['label'],
        'type': case.split('_')[0]
    }


# --- Plotting and Animation ---
fig = plt.figure(figsize=(8, 8),facecolor='grey')
ax = fig.add_subplot(111, projection='polar')

# Plot the Black Hole (Event Horizon)
ax.fill_between(np.linspace(0, 2*np.pi, 100), 0, R_S, color='black', label=f'Event Horizon (r={R_S/M:.0f}M)')

# Plot the Photon Sphere
ax.plot(np.linspace(0, 2*np.pi, 100), [R_PS]*100, 'r--', label=f'Photon Sphere (r={R_PS/M:.0f}M)')

# Plot faint full trajectories
for traj in trajectories.values():
    ax.plot(traj['phi'], traj['r'], color=traj['color'], alpha=0.3, linewidth=0.5, label=traj['label'])

# Uniform speed in distance per frame
uniform_speed = 1

# Initialize photon dots - one artist per type with smaller size
dots = []
for traj_type, color in [('scatter', 'blue'), ('capture', 'red'), ('critical', 'black')]:
    dot, = ax.plot([], [], 'o', color=color, markersize=3, linestyle='none')  # Reduced size
    dots.append(dot)

# Animation function (unchanged, included for completeness)
num_photons_per_trajectory = 100
spacing = 4

def animate(frame):
    scatter_phi, scatter_r = [], []
    capture_phi, capture_r = [], []
    critical_phi, critical_r = [], []

    for photon_num in range(num_photons_per_trajectory):
        photon_frame = frame - (photon_num * spacing)
        if photon_frame < 0:
            continue
        arc_pos = photon_frame * uniform_speed

        for case, traj in trajectories.items():
            if arc_pos > traj['max_arc_length']:
                continue
            r_val = np.interp(arc_pos, traj['arc_length'], traj['r'])
            phi_val = np.interp(arc_pos, traj['arc_length'], traj['phi'])

            if traj['type'] == 'scatter':
                scatter_phi.append(phi_val)
                scatter_r.append(r_val)
            elif traj['type'] == 'capture':
                capture_phi.append(phi_val)
                capture_r.append(r_val)
            elif traj['type'] == 'critical':
                critical_phi.append(phi_val)
                critical_r.append(r_val)

    dots[0].set_data(scatter_phi, scatter_r) if scatter_r else dots[0].set_data([], [])
    dots[1].set_data(capture_phi, capture_r) if capture_r else dots[1].set_data([], [])
    dots[2].set_data(critical_phi, critical_r) if critical_r else dots[2].set_data([], [])

    return dots

# Rest of the animation setup (unchanged)
max_length = max(traj['max_arc_length'] for traj in trajectories.values())
frames_for_longest = np.ceil(max_length / uniform_speed)
max_frames = int(frames_for_longest + (num_photons_per_trajectory - 1) * spacing) + 1

anim = FuncAnimation(fig, animate, frames=max_frames, interval=20, blit=True, repeat=True)

ax.set_facecolor("white")
ax.set_ylim(0, 50)
ax.set_rlabel_position(90)
ax.set_title(f'Photon Trajectories in Schwarzschild Spacetime (M={M})', fontsize=16, color='white')
ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.show()
