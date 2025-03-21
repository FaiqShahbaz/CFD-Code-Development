import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

def build_up_b(rho, dt, dx, dy, u, v):
    """
    Build the source term for the pressure Poisson equation for channel flow.
    Periodic boundary conditions in the x-direction are applied.
    """
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (
        1/dt * ((u[1:-1, 2:] - u[1:-1, :-2])/(2*dx) +
                (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy))
        - ((u[1:-1, 2:] - u[1:-1, :-2])/(2*dx))**2
        - 2 * ((u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy) *
               (v[1:-1, 2:] - v[1:-1, :-2])/(2*dx))
        - ((v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy))**2
    ))
    # Periodic BC Pressure @ x = 2 (last column)
    b[1:-1, -1] = (rho * (
        1/dt * ((u[1:-1, 0] - u[1:-1, -2])/(2*dx) +
                (v[2:, -1] - v[:-2, -1])/(2*dy))
        - ((u[1:-1, 0] - u[1:-1, -2])/(2*dx))**2
        - 2 * ((u[2:, -1] - u[:-2, -1])/(2*dy) *
               (v[1:-1, 0] - v[1:-1, -2])/(2*dx))
        - ((v[2:, -1] - v[:-2, -1])/(2*dy))**2
    ))
    # Periodic BC Pressure @ x = 0 (first column)
    b[1:-1, 0] = (rho * (
        1/dt * ((u[1:-1, 1] - u[1:-1, -1])/(2*dx) +
                (v[2:, 0] - v[:-2, 0])/(2*dy))
        - ((u[1:-1, 1] - u[1:-1, -1])/(2*dx))**2
        - 2 * ((u[2:, 0] - u[:-2, 0])/(2*dy) *
               (v[1:-1, 1] - v[1:-1, -1])/(2*dx))
        - ((v[2:, 0] - v[:-2, 0])/(2*dy))**2
    ))
    return b

def pressure_poisson_periodic(p, dx, dy, b, nit):
    """
    Solve the pressure Poisson equation with periodic boundary conditions in x 
    for channel flow. Wall (Neumann) BCs are applied in y.
    """
    for q in range(nit):
        pn = p.copy()
        # Update interior points
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) *
                         b[1:-1, 1:-1])
        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2]) * dy**2 +
                        (pn[2:, -1] + pn[:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) *
                       b[1:-1, -1])
        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2 +
                       (pn[2:, 0] + pn[:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      (dx**2 * dy**2) / (2 * (dx**2 + dy**2)) *
                      b[1:-1, 0])
        # Wall BC (Neumann) at y = 0 and y = 2
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
    return p

def channel_flow(u, v, p, dx, dy, dt, rho, nu, F, nit, tol=1e-3, max_steps=10000):
    """
    Solve the channel flow problem with periodic BCs in x and wall BCs in y.
    
    Returns:
        u, v, p: Updated velocity and pressure fields.
        stepcount: Number of iterations performed.
    """
    udiff = 1.0
    stepcount = 0

    while udiff > tol and stepcount < max_steps:
        un = u.copy()
        vn = v.copy()

        # Build source term and solve for pressure with periodic BCs
        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(p, dx, dy, b, nit)

        # Update u-velocity (interior)
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])) +
                         F * dt)
        # Update v-velocity (interior)
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt/(2*rho*dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
        # Apply periodic BC for u in x-direction
        u[1:-1, -1] = (un[1:-1, -1] -
                       un[1:-1, -1] * dt/dx * (un[1:-1, -1] - un[1:-1, -2]) -
                       vn[1:-1, -1] * dt/dy * (un[1:-1, -1] - un[:-2, -1]) -
                       dt/(2*rho*dx) * (p[1:-1, 0] - p[1:-1, -2]) +
                       nu * (dt/dx**2 * (un[1:-1, 0] - 2*un[1:-1, -1] + un[1:-1, -2]) +
                             dt/dy**2 * (un[2:, -1] - 2*un[1:-1, -1] + un[:-2, -1])) +
                       F * dt)
        u[1:-1, 0] = (un[1:-1, 0] -
                      un[1:-1, 0] * dt/dx * (un[1:-1, 0] - un[1:-1, -1]) -
                      vn[1:-1, 0] * dt/dy * (un[1:-1, 0] - un[:-2, 0]) -
                      dt/(2*rho*dx) * (p[1:-1, 1] - p[1:-1, -1]) +
                      nu * (dt/dx**2 * (un[1:-1, 1] - 2*un[1:-1, 0] + un[1:-1, -1]) +
                            dt/dy**2 * (un[2:, 0] - 2*un[1:-1, 0] + un[:-2, 0])) +
                      F * dt)
        # Apply periodic BC for v in x-direction
        v[1:-1, -1] = (vn[1:-1, -1] -
                       un[1:-1, -1] * dt/dx * (vn[1:-1, -1] - vn[1:-1, -2]) -
                       vn[1:-1, -1] * dt/dy * (vn[1:-1, -1] - vn[:-2, -1]) -
                       dt/(2*rho*dy) * (p[2:, -1] - p[:-2, -1]) +
                       nu * (dt/dx**2 * (vn[1:-1, 0] - 2*vn[1:-1, -1] + vn[1:-1, -2]) +
                             dt/dy**2 * (vn[2:, -1] - 2*vn[1:-1, -1] + vn[:-2, -1])))
        v[1:-1, 0] = (vn[1:-1, 0] -
                      un[1:-1, 0] * dt/dx * (vn[1:-1, 0] - vn[1:-1, -1]) -
                      vn[1:-1, 0] * dt/dy * (vn[1:-1, 0] - vn[:-2, 0]) -
                      dt/(2*rho*dy) * (p[2:, 0] - p[:-2, 0]) +
                      nu * (dt/dx**2 * (vn[1:-1, 1] - 2*vn[1:-1, 0] + vn[1:-1, -1]) +
                            dt/dy**2 * (vn[2:, 0] - 2*vn[1:-1, 0] + vn[:-2, 0])))
        # Apply wall BC: u, v = 0 at y = 0 and y = 2
        u[0, :], u[-1, :] = 0, 0
        v[0, :], v[-1, :] = 0, 0

        # Compute relative change in u for convergence
        udiff = np.abs(np.sum(u) - np.sum(un)) / (np.abs(np.sum(u)) + 1e-10)
        stepcount += 1

    return u, v, p, stepcount

def plot_results(X, Y, u, v):
    """
    Plot the primary velocity field using a quiver plot.
    """
    plt.figure(figsize=(11, 7), dpi=100)
    plt.quiver(X, Y, u, v)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Channel Flow - Velocity Field')
    plt.savefig("channel_flow_velocity_quiver.png", bbox_inches='tight')
    plt.show()

def plot_extra_results(X, Y, u, v, p, dx, dy, smoothing_sigma=1, contour_levels=50):
    """
    Plot additional extracted quantities with improved smoothness:
    - Smoothed Velocity Magnitude
    - Smoothed Vorticity Field
    - Centerline Velocity Profile (u vs. Y at mid x)
    """
    # Compute Velocity Magnitude and smooth it
    vel_mag = np.sqrt(u**2 + v**2)
    vel_mag_smoothed = gaussian_filter(vel_mag, sigma=smoothing_sigma)
    
    plt.figure(figsize=(11,7), dpi=100)
    contour = plt.contourf(X, Y, vel_mag_smoothed, levels=contour_levels, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(contour)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Magnitude')
    plt.savefig("channel_flow_velocity_magnitude_smoothed.png", bbox_inches='tight')
    plt.show()

    # Compute Vorticity Field and smooth it
    vort = np.zeros_like(u)
    vort[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, :-2])/(2*dx) -
                        (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy))
    vort_smoothed = gaussian_filter(vort, sigma=smoothing_sigma)
    
    plt.figure(figsize=(11,7), dpi=100)
    contour = plt.contourf(X, Y, vort_smoothed, levels=contour_levels, alpha=0.5, cmap=cm.coolwarm)
    plt.colorbar(contour)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vorticity Field')
    plt.savefig("channel_flow_vorticity_smoothed.png", bbox_inches='tight')
    plt.show()

    # Centerline Velocity Profile (u vs. Y at mid x)
    mid_x = X.shape[1] // 2
    center_u = u[:, mid_x]
    y_center = Y[:, mid_x]
    plt.figure(figsize=(8,6), dpi=100)
    plt.plot(center_u, y_center, 'b.-')
    plt.xlabel('u-velocity')
    plt.ylabel('Y')
    plt.title('Centerline Velocity Profile (u vs Y at mid x)')
    plt.grid(True)
    plt.savefig("channel_flow_centerline_velocity.png", bbox_inches='tight')
    plt.show()

def main():
    # Domain parameters
    nx, ny = 41, 41
    x_start, x_end = 0, 2
    y_start, y_end = 0, 2
    x = np.linspace(x_start, x_end, nx)
    y = np.linspace(y_start, y_end, ny)
    X, Y = np.meshgrid(x, y)
    dx = (x_end - x_start) / (nx - 1)
    dy = (y_end - y_start) / (ny - 1)
    
    # Simulation parameters
    dt = 0.01
    rho = 1
    nu = 0.1
    F = 1      # Forcing term
    nit = 50   # Number of iterations for pressure Poisson
    
    # Initialize fields: u and v as zeros; p as ones.
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx))
    
    # Run channel flow simulation
    u, v, p, steps = channel_flow(u, v, p, dx, dy, dt, rho, nu, F, nit)
    print("Steps to convergence:", steps)
    
    # Plot primary velocity field (quiver)
    plot_results(X, Y, u, v)
    # Plot additional extracted fields: velocity magnitude, vorticity, and centerline profile
    plot_extra_results(X, Y, u, v, p, dx, dy, smoothing_sigma=1, contour_levels=50)

if __name__ == '__main__':
    main()
