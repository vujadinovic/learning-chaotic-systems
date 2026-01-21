import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import matplotlib.animation as animation

np.random.seed(11231)

G = 9.8 
L1 = 1.0 
L2 = 1.0 
L = L1 + L2 
M1 = 1.0 
M2 = 1.0 
t_stop = 2.5 
dt = 0.01
num_simulations = 100

def derivs(t, state):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)
    dydx[2] = state[3]
    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)
    return dydx

def get_acc(state):
    d = derivs(0, state)
    return np.array([d[1], d[3]])

def state_to_momenta(state):
    th1, w1, th2, w2 = state
    p1 = (M1 + M2) * L1**2 * w1 + M2 * L1 * L2 * w2 * cos(th1 - th2)
    p2 = M2 * L2**2 * w2 + M2 * L1 * L2 * w1 * cos(th1 - th2)
    return np.array([p1, p2])

t = np.arange(0, t_stop, dt)

all_ode_data = []
all_hnn_data = []

last_y = None 

print(f"Generating {num_simulations} simulations using Velocity Verlet...")

for sim_id in range(num_simulations):
    th1_init = np.random.uniform(-np.pi, np.pi)
    th2_init = np.random.uniform(-np.pi, np.pi)
    w1_init = np.random.uniform(-2.0, 2.0)
    w2_init = np.random.uniform(-2.0, 2.0)
    
    state = np.array([th1_init, w1_init, th2_init, w2_init])
    
    y = np.zeros((len(t), 4))
    y[0] = state
    
    for i in range(len(t) - 1):
        curr_q = np.array([y[i, 0], y[i, 2]])
        curr_v = np.array([y[i, 1], y[i, 3]])
        
        a_t = get_acc(y[i])
        
        next_q = curr_q + curr_v * dt + 0.5 * a_t * dt**2
        
        pred_state = np.array([next_q[0], curr_v[0], next_q[1], curr_v[1]])
        a_next_pred = get_acc(pred_state)
        next_v = curr_v + 0.5 * (a_t + a_next_pred) * dt
        
        y[i+1] = np.array([next_q[0], next_v[0], next_q[1], next_v[1]])
    
    if sim_id == num_simulations - 1:
        last_y = y

    ids = np.full((len(t), 1), sim_id)
    
    ode_chunk = np.column_stack((ids, t, y))
    all_ode_data.append(ode_chunk)
    
    momenta = np.array([state_to_momenta(s) for s in y])
    q_vals = y[:, [0, 2]]
    p_vals = momenta
    
    dq_vals = np.gradient(q_vals, dt, axis=0)
    dp_vals = np.gradient(p_vals, dt, axis=0)
    
    hnn_chunk = np.column_stack((ids, t, q_vals, p_vals, dq_vals, dp_vals))
    all_hnn_data.append(hnn_chunk)

final_ode_data = np.vstack(all_ode_data)
ode_header = "id,t,th1,w1,th2,w2"
np.savetxt("ode_data_100sims.csv", final_ode_data, delimiter=",", header=ode_header, comments='')

final_hnn_data = np.vstack(all_hnn_data)
hnn_header = "id,t,q1,q2,p1,p2,dq1,dq2,dp1,dp2"
np.savetxt("hnn_data_100sims.csv", final_hnn_data, delimiter=",", header=hnn_header, comments='')

print("Data generation complete. Saved 'ode_data_100sims.csv' and 'hnn_data_100sims.csv'.")

x1 = L1*sin(last_y[:, 0])
y1 = -L1*cos(last_y[:, 0])
x2 = L2*sin(last_y[:, 2]) + x1
y2 = -L2*cos(last_y[:, 2]) + y1

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    history_x = x2[:i]
    history_y = y2[:i]
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text

ani = animation.FuncAnimation(fig, animate, len(last_y), interval=dt*1000, blit=True)
plt.show()
