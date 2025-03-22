from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import os, time

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Domain and simulation parameters (identical to serial)
Lx, Ly = 1.0, 1.0
Nx, Ny = 100, 100
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 1e-5
Nt = 50000
vx, vy = 0.0, 0.0
D = 1.0

# Boundary conditions (Dirichlet)
u_x0, u_xL = 0.0, 1.0
u_y0, u_yL = 0.0, 1.0

# Stability check
beta_x = D * dt / dx**2
beta_y = D * dt / dy**2
if rank == 0 and (beta_x + beta_y > 0.5):
    print(f"WARNING: Unstable! βx+βy = {beta_x + beta_y:.2f}")

# Domain decomposition (assumes square or near-square process grid)
px = int(np.floor(np.sqrt(size)))
while size % px != 0:
    px -= 1
py = size // px
if rank == 0:
    print(f"MPI Decomposition: {px} x {py} processes")

ix = rank % px
iy = rank // px
local_Nx = Nx // px
local_Ny = Ny // py
x_start = ix * local_Nx
y_start = iy * local_Ny

# Local grid
x_local = np.linspace(x_start * dx, (x_start + local_Nx - 1) * dx, local_Nx)
y_local = np.linspace(y_start * dy, (y_start + local_Ny - 1) * dy, local_Ny)
X_local, Y_local = np.meshgrid(x_local, y_local, indexing='ij')

# Initial condition
u_local = np.sin(np.pi * X_local / Lx) * np.sin(np.pi * Y_local / Ly)

# Time-stepping loop (measure time only on rank 0)
if rank == 0:
    start_time = time.time()

for n in range(Nt):
    u_new = u_local.copy()
    for i in range(1, local_Nx - 1):
        for j in range(1, local_Ny - 1):
            u_new[i, j] = (
                u_local[i, j]
                - vx * dt / (2 * dx) * (u_local[i+1, j] - u_local[i-1, j])
                - vy * dt / (2 * dy) * (u_local[i, j+1] - u_local[i, j-1])
                + D * dt * (
                    (u_local[i+1, j] - 2*u_local[i, j] + u_local[i-1, j]) / dx**2 +
                    (u_local[i, j+1] - 2*u_local[i, j] + u_local[i, j-1]) / dy**2
                )
            )

    # MPI boundary exchange with contiguous buffers
    left = np.empty(local_Ny, dtype=np.float64)
    right = np.empty(local_Ny, dtype=np.float64)
    top = np.empty(local_Nx, dtype=np.float64)
    bottom = np.empty(local_Nx, dtype=np.float64)

    if ix > 0:
        comm.Sendrecv(np.ascontiguousarray(u_local[1, :], dtype=np.float64),
                      dest=rank - 1, recvbuf=left)
        u_new[0, :] = left
    if ix < px - 1:
        comm.Sendrecv(np.ascontiguousarray(u_local[-2, :], dtype=np.float64),
                      dest=rank + 1, recvbuf=right)
        u_new[-1, :] = right

    if iy > 0:
        comm.Sendrecv(np.ascontiguousarray(u_local[:, 1], dtype=np.float64),
                      dest=rank - px, recvbuf=bottom)
        u_new[:, 0] = bottom
    if iy < py - 1:
        comm.Sendrecv(np.ascontiguousarray(u_local[:, -2], dtype=np.float64),
                      dest=rank + px, recvbuf=top)
        u_new[:, -1] = top

    # Apply Dirichlet boundary conditions
    if x_start == 0:
        u_new[0, :] = u_x0
    if x_start + local_Nx == Nx:
        u_new[-1, :] = u_xL
    if y_start == 0:
        u_new[:, 0] = u_y0
    if y_start + local_Ny == Ny:
        u_new[:, -1] = u_yL

    u_local = u_new.copy()

if rank == 0:
    end_time = time.time()
    print(f"Parallel simulation time: {end_time - start_time:.2f} seconds")

# Gather results from all processes to root
u_global = None
if rank == 0:
    u_global = np.empty((size, local_Nx, local_Ny), dtype=np.float64)
comm.Gather(u_local, u_global, root=0)

# Stitch the subdomains and plot on root
if rank == 0:
    full = np.zeros((Nx, Ny))
    for r in range(size):
        ix = r % px
        iy = r // px
        full[ix*local_Nx:(ix+1)*local_Nx, iy*local_Ny:(iy+1)*local_Ny] = u_global[r]

    X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny), indexing='ij')
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, full, 50, cmap='viridis')
    plt.colorbar(label='u(x, y, T)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Advection-Diffusion with MPI')
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/solution_2d_mpi.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Parallel solution saved: results/solution_2d_mpi.png")
