import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def build_up_b(rho, dt, u, v, dx, dy):
    """
    Compute the source term for the pressure Poisson equation.
    
    Parameters:
        rho : float
            Fluid density.
        dt : float
            Time step.
        u, v : 2D numpy arrays
            Velocity components.
        dx, dy : float
            Spatial step sizes.
    
    Returns:
        b : 2D numpy array
            The computed source term.
    """
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (
        rho * (
            1 / dt * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
                      + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))
            - ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2
            - 2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
                   * (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx))
            - ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2
        )
    )
    return b

def pressure_poisson(p, dx, dy, b, nit, tol=1e-4):
    """
    Solve the pressure Poisson equation using an iterative method.
    
    Parameters:
        p : 2D numpy array
            Initial pressure field.
        dx, dy : float
            Spatial step sizes.
        b : 2D numpy array
            Source term computed from build_up_b.
        nit : int
            Maximum number of iterations.
        tol : float, optional
            Tolerance for early convergence.
    
    Returns:
        p : 2D numpy array
            The updated pressure field.
    """
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2
             + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2)
            / (2 * (dx**2 + dy**2))
            - dx**2 * dy**2 / (2 * (dx**2 + dy**2))
            * b[1:-1, 1:-1]
        )
        # Apply boundary conditions
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2

        # Check for convergence (optional)
        if np.linalg.norm(p - pn, ord=2) < tol:
            break

    return p

def cavity_flow(nt, u, v, p, dx, dy, dt, rho, nu, nit):
    """
    Solve the cavity flow problem over a given number of time steps.
    
    Parameters:
        nt : int
            Number of time steps.
        u, v : 2D numpy arrays
            Velocity fields.
        p : 2D numpy array
            Pressure field.
        dx, dy : float
            Spatial step sizes.
        dt : float
            Time step.
        rho : float
            Fluid density.
        nu : float
            Kinematic viscosity.
        nit : int
            Number of iterations for the pressure solver.
    
    Returns:
        u, v, p : 2D numpy arrays
            Updated velocity and pressure fields.
    """
    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        # Compute source term for pressure Poisson equation
        b = build_up_b(rho, dt, un, vn, dx, dy)
        # Update pressure field
        p = pressure_poisson(p, dx, dy, b, nit)

        # Update velocity field using finite difference approximations
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu * (dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])
                    + dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1])
            + nu * (dt / dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])
                    + dt / dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
        )

        # Enforce boundary conditions for velocity (cavity flow conditions)
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # moving lid with constant velocity
        v[0, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        v[-1, :] = 0

    return u, v, p

def plot_results(X, Y, u, v, p):
    """
    Plot the pressure field and velocity vectors (quiver and streamlines).
    The titles have been simplified, and black contour lines are removed.
    """

    # --- Quiver Plot ---
    plt.figure(figsize=(11, 7), dpi=100)
    # Filled contours only (no black outlines)
    contour = plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(contour)
    
    # Quiver plot for velocity
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cavity Flow - Velocity Field (Quiver)')
    plt.savefig("cavity_flow_velocity_quiver.png", bbox_inches='tight')
    plt.show()

    # --- Streamlines Plot ---
    plt.figure(figsize=(11, 7), dpi=100)
    # Filled contours only
    contour = plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar(contour)
    
    # Streamlines for velocity
    plt.streamplot(X, Y, u, v, color='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cavity Flow - Streamlines')
    plt.savefig("cavity_flow_streamlines.png", bbox_inches='tight')
    plt.show()

def main():
    # Domain parameters
    nx, ny = 100, 100
    x_start, x_end = 0, 2
    y_start, y_end = 0, 2
    x = np.linspace(x_start, x_end, nx)
    y = np.linspace(y_start, y_end, ny)
    X, Y = np.meshgrid(x, y)
    dx = (x_end - x_start) / (nx - 1)
    dy = (y_end - y_start) / (ny - 1)

    # Simulation parameters
    nt = 1000    # Number of time steps
    nit = 50     # Maximum iterations for pressure Poisson
    rho = 1.0    # Fluid density
    nu = 0.1     # Kinematic viscosity
    dt = 0.001   # Time step

    # Initialize fields
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Run cavity flow simulation
    u, v, p = cavity_flow(nt, u, v, p, dx, dy, dt, rho, nu, nit)

    # Plot and save the results
    plot_results(X, Y, u, v, p)

if __name__ == '__main__':
    main()
