import numpy as np
from scipy.integrate import solve_ivp

def lorenz_system(t, y, sigma, rho, beta):
    """
    Defines the Lorenz system of ordinary differential equations.

    Parameters:
        t: Current time.
        y: A 1D NumPy array containing the current state [x, y, z].
        sigma, rho, beta: Parameters of the Lorenz system.

    Returns:
        dydt: A 1D NumPy array representing the derivative of the state.
    """
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    dydt = [dxdt, dydt, dzdt]
    return dydt

# Set the parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial conditions
initial_state = [1.0, 0.0, 0.0]

# Time span for simulation
t_span = (0, 40)

# Time points at which the solution will be recorded
t_eval = np.linspace(0, 40, 10000)

# Solve the Lorenz system
sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

# Extract the solution
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

# Plot the solution, e.g., using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(sol.t, x, label='x(t)')
plt.plot(sol.t, y, label='y(t)')
plt.plot(sol.t, z, label='z(t)')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Lorenz System Simulation')
plt.legend()
plt.grid(True)
plt.show()
