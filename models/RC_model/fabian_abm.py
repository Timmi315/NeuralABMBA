import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Initial condition
y0 = 290  # Initial value of y at t = 0
R = 100  # Resistance (ohms)
C = 3  # Capacitance (Farads)
t_start = 0
t_end = 5*24*60  # in min
time_step = 1 # in min
# random Q
Q_low = 0
Q_up = 0
plotting = True

def Q_hvac(t):
    Q = np.random.uniform(Q_low, Q_up)
    return Q if t >= 0 else 0  # Step input of 5V

def rc_circuit(t, Vc, R, C):
    Vin = 270 #+ 5*np.sin(2 * np.pi / 24 /60 * t)
    Q_h = Q_hvac(t)
    dVcdt = (Vin - Vc) / (R * C) + Q_h/C
    return dVcdt

# Define the objective function for least squares
def objective(params, t, y_data):
    R, C = params
    sol = solve_ivp(rc_circuit, [t[0], t[-1]], [y_data[0]], t_eval=t, args=(R, C))
    y_pred = sol.y[0]
    return np.sum((y_data - y_pred) ** 2)


t_span = np.arange(t_start, t_end + time_step, time_step)

# Solve the ODE using scipy.integrate.solve_ivp
solution = solve_ivp(rc_circuit, [t_start, t_end], [y0], args=( R, C), t_eval=t_span)
meas_data = solution.y[0]
print(meas_data.shape)

# Plot the solution
if plotting==True:
    plt.plot(solution.t, meas_data, label='Voltage across Capacitor (Vc)')
    plt.xlabel('Time')
    plt.ylabel('Solution y')
    plt.title('Solution of ODE with External Input using Euler Method')
    plt.legend()
    plt.grid(True)
    plt.show()

initial_params = [35, 2]

#exit()
# Optimize parameters 
result = minimize(objective, initial_params, args=(solution.t, meas_data), method='Nelder-Mead')

# Extract optimized parameters
optimal_params = result.x
optimal_R, optimal_C = optimal_params

print("Optimal R:", optimal_R)
print("Optimal C:", optimal_C)

