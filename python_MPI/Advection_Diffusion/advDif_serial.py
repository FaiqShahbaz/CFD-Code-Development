import numpy as np
import matplotlib.pyplot as plt
import os, time

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Time-stepping parameters
dt = 1e-5
Nt = 50000

# Physical parameters
vx, vy = 0.0, 0.0
D = 1.0

# Boundary conditions (Dirichlet)
u_x0, u_xL = 0.0, 1.0
u_y0, u_yL = 0.0, 1.0

# Stability check
beta_x = D * dt / dx**2
beta_y = D * dt / dy**2
if beta_x + beta_y > 0.5:
    print(f"WARNING: Unstable! βx+βy = {beta_x + beta_y:.2f}")

# Create grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition: sine function product
u = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

# Apply Dirichlet boundary conditions
u[0, :] = u_x0
u[-1, :] = u_xL
u[:, 0] = u_y0
u[:, -1] = u_yL

# Time-stepping loop (measure execution time)
start_time = time.time()
for n in range(Nt):
    u_new = u.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u_new[i, j] = (u[i, j]
                           - vx * dt / (2*dx) * (u[i+1, j] - u[i-1, j])
                           - vy * dt / (2*dy) * (u[i, j+1] - u[i, j-1])
                           + D * dt * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2
                                       + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2))
    # Reapply Dirichlet BCs (if necessary)
    u_new[0, :] = u_x0
    u_new[-1, :] = u_xL
    u_new[:, 0] = u_y0
    u_new[:, -1] = u_yL
    u = u_new.copy()
end_time = time.time()
print(f"Serial simulation time: {end_time - start_time:.2f} seconds")

# Plot and save the final solution
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u, 50, cmap='viridis')
plt.colorbar(label='u(x, y, T)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Advection-Diffusion (Serial)')
plt.savefig("results/solution_2d_serial.png", dpi=300, bbox_inches='tight')
plt.close()
print("Serial solution saved: results/solution_2d_serial.png")
