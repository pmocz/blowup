import jax.numpy as jnp
import matplotlib.pyplot as plt

"""
Burgers 1D

Philip Mocz (2025)

Solve the 1D inviscid Burgers (periodic) equation:
u_t + u u_x = 0
"""


def burgers_1d_step(u, dx, dt):
    """Perform a single time step of the 1D inviscid Burgers equation using upwind scheme."""
    # Compute flux
    flux = 0.5 * u**2

    # Compute numerical fluxes at cell interfaces
    flux_right = jnp.roll(flux, -1)
    flux_left = jnp.roll(flux, 1)

    # Upwind scheme
    # du_dt = jnp.where(u > 0, (flux - flux_left), (flux_right - flux))

    # Local Lax-Friedrichs (Rusanov) scheme
    a = jnp.abs(u)
    du_dt_right = 0.5 * (flux_right + flux) - 0.5 * a * (jnp.roll(u, -1) - u)
    du_dt_left = 0.5 * (flux + flux_left) - 0.5 * a * (u - jnp.roll(u, 1))
    du_dt = du_dt_right - du_dt_left

    # Update solution
    u_new = u - dt / dx * du_dt
    return u_new


def simulate_burgers_1d(u0, dx, dt, num_steps):
    """Simulate the 1D inviscid Burgers equation over a number of time steps."""
    u = u0
    for _ in range(num_steps):
        u = burgers_1d_step(u, dx, dt)
    return u


# Example usage
if __name__ == "__main__":
    # Initial condition: a sine wave
    x = jnp.linspace(0, 2 * jnp.pi, 400)
    u0 = jnp.sin(x)

    # Simulation parameters
    dx = x[1] - x[0]
    dt = 0.01
    num_steps = 100

    # Run simulation
    u_final = simulate_burgers_1d(u0, dx, dt, num_steps)

    # Plot final result
    plt.plot(x, u_final)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.show()
