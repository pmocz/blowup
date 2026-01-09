import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from matplotlib import cm
import time

"""
Search for Self-Similar solution in the Boussinesq 2D equation

Philip Mocz (2025)

(Omega, U1, Phi) odd in y1
(U2, Psi) even in y1

halfplane: (y1,y2) in [-20,20] x [0, 20]

U2(y1,0) = 0
d Omega(0,0) / d y1 = -1

TODO: XXX NOT YET WORKING!
"""


def grad_y1(F, dy1):
    """Compute gradient in y1 direction using central differences."""
    dF_dy1 = jnp.zeros_like(F)
    dF_dy1 = dF_dy1.at[1:-1, :].set((F[2:, :] - F[:-2, :]) / (2 * dy1))
    dF_dy1 = dF_dy1.at[0, :].set((-3 * F[0, :] + 4 * F[1, :] - F[2, :]) / (2 * dy1))
    dF_dy1 = dF_dy1.at[-1, :].set((3 * F[-1, :] - 4 * F[-2, :] + F[-3, :]) / (2 * dy1))
    return dF_dy1


def grad_y2(F, dy2):
    """Compute gradient in y2 direction using central differences."""
    dF_dy2 = jnp.zeros_like(F)
    dF_dy2 = dF_dy2.at[:, 1:-1].set((F[:, 2:] - F[:, :-2]) / (2 * dy2))
    dF_dy2 = dF_dy2.at[:, 0].set((-3 * F[:, 0] + 4 * F[:, 1] - F[:, 2]) / (2 * dy2))
    dF_dy2 = dF_dy2.at[:, -1].set((3 * F[:, -1] - 4 * F[:, -2] + F[:, -3]) / (2 * dy2))
    return dF_dy2


def curl(U1, U2, dy1, dy2):
    """Compute the curl of the velocity field."""
    dU2_dy1 = grad_y1(U2, dy1)
    dU1_dy2 = grad_y2(U1, dy2)
    return dU2_dy1 - dU1_dy2


def div(U1, U2, dy1, dy2):
    """Compute the divergence of the velocity field."""
    dU1_dy1 = grad_y1(U1, dy1)
    dU2_dy2 = grad_y2(U2, dy2)
    return dU1_dy1 + dU2_dy2


def convert_to_reduced(U1, U2, Phi, Psi, dy1, dy2):
    """Convert full fields to reduced fields using symmetry."""
    # U1, Phi odd in y1
    # U2, Psi even in y1
    # U2(y1,0) = 0
    # odds:
    U1_r = U1[U1.shape[0] // 2 + 1 :, :]
    Phi_r = Phi[Phi.shape[0] // 2 + 1 :, :]
    # evens:
    U2_r = U2[U2.shape[0] // 2 :, 1:]  # remove y2=0 column
    Psi_r = Psi[Psi.shape[0] // 2 :, :]

    # d Omega(0,0) / d y1 = -1
    # => d_xy U1 = 1
    b = U1_r[0, 1]
    c = U1_r[0, 2]
    a = (4 * b - c - 2 * dy1 * dy2) / 3
    U1_r = U1_r.at[0, 0].set(a)
    return U1_r, U2_r, Phi_r, Psi_r


def convert_from_reduced(U1_r, U2_r, Phi_r, Psi_r, dy1, dy2):
    """Convert reduced fields back to full fields using symmetry."""
    # U1, Phi odd in y1
    # U2, Psi even in y1
    # U2(y1,0) = 0

    # d Omega(0,0) / d y1 = -1
    # => d_xy U1 = 1
    b = U1_r[0, 1]
    c = U1_r[0, 2]
    a = (4 * b - c - 2 * dy1 * dy2) / 3
    U1_r = U1_r.at[0, 0].set(a)

    # odds:
    U1_top = -jnp.flipud(U1_r)
    U1 = jnp.vstack([U1_top, jnp.zeros((1, U1_r.shape[1])), U1_r])
    Phi_top = -jnp.flipud(Phi_r)
    Phi = jnp.vstack([Phi_top, jnp.zeros((1, Phi_r.shape[1])), Phi_r])
    # evens:
    U2_top = jnp.flipud(U2_r[1:, :])  # remove y2=0 row
    U2 = jnp.vstack([U2_top, U2_r])
    U2 = jnp.hstack([jnp.zeros((U2.shape[0], 1)), U2])  # add zero row at y2=0 for U2
    Psi_top = jnp.flipud(Psi_r[1:, :])
    Psi = jnp.vstack([Psi_top, Psi_r])

    Omega = curl(U1, U2, dy1, dy2)

    return U1, U2, Phi, Psi, Omega


@jax.jit
def residual(U, Y1, Y2):
    """Compute the residual of the discretized PDE."""
    lambda_, U1_r, U2_r, Phi_r, Psi_r = U
    dy1 = Y1[1, 0] - Y1[0, 0]
    dy2 = Y2[0, 1] - Y2[0, 0]

    U1, U2, Phi, Psi, Omega = convert_from_reduced(U1_r, U2_r, Phi_r, Psi_r, dy1, dy2)

    dy1_Omega = grad_y1(Omega, dy1)
    dy2_Omega = grad_y2(Omega, dy2)
    dy1_Psi = grad_y1(Psi, dy1)
    dy2_Psi = grad_y2(Psi, dy2)
    dy1_Phi = grad_y1(Phi, dy1)
    dy2_Phi = grad_y2(Phi, dy2)
    dy1_U1 = grad_y1(U1, dy1)
    dy2_U1 = grad_y2(U1, dy2)
    dy1_U2 = grad_y1(U2, dy1)
    dy2_U2 = grad_y2(U2, dy2)

    fac1 = (1 + lambda_) * Y1 + U1
    fac2 = (1 + lambda_) * Y2 + U2

    eqn1 = Omega + fac1 * dy1_Omega + fac2 * dy2_Omega - Phi
    eqn2 = (2 + dy1_U1) * Phi + fac1 * dy1_Phi + fac2 * dy2_Phi + dy1_U2 * Psi
    eqn3 = (2 + dy2_U2) * Psi + fac1 * dy1_Psi + fac2 * dy2_Psi + dy2_U1 * Phi
    eqn4 = div(U1, U2, dy1, dy2)
    eqn5 = dy1_Psi - dy2_Phi

    error_norm = (
        jnp.linalg.norm(eqn1)
        + jnp.linalg.norm(eqn2)
        + jnp.linalg.norm(eqn3)
        + jnp.linalg.norm(eqn4)
        + jnp.linalg.norm(eqn5)
    )
    return error_norm


def search(
    Y1,
    Y2,
    lambda_init,
    U1_init,
    U2_init,
    Phi_init,
    Psi_init,
    num_iters=10000,  # XXX 1000,
    lr=0.001,
):
    """Find the self-similar solution U(y) using Adam optimizer."""

    dy1 = Y1[1, 0] - Y1[0, 0]
    dy2 = Y2[0, 1] - Y2[0, 0]

    U1_r, U2_r, Phi_r, Psi_r = convert_to_reduced(
        U1_init, U2_init, Phi_init, Psi_init, dy1, dy2
    )
    U = lambda_init, U1_r, U2_r, Phi_r, Psi_r

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(U)

    for i in range(num_iters):
        res_grad = jax.grad(residual)(U, Y1, Y2)
        updates, opt_state = optimizer.update(res_grad, opt_state)
        U = optax.apply_updates(U, updates)
        print("Iteration:", i)
        print("  residual norm:", residual(U, Y1, Y2))
    lambda_, U1_r, U2_r, Phi_r, Psi_r = U
    U1, U2, Phi, Psi, Omega = convert_from_reduced(U1_r, U2_r, Phi_r, Psi_r, dy1, dy2)
    return lambda_, U1, U2, Omega, Phi, Psi


def main():
    """Search and plot results."""
    # Grid
    nhalf = 16  # XXX 200
    y1lin = jnp.linspace(-20, 20, 2 * nhalf + 1)
    y2lin = jnp.linspace(0, 20, nhalf + 1)
    Y1, Y2 = jnp.meshgrid(y1lin, y2lin, indexing="ij")
    # Initial guess
    lambda_init = 1.917
    U1_init = -jnp.sin(0.5 * jnp.pi / 20.0 * Y1) * 20
    # Omega_init = -jnp.sin(1.5 * jnp.pi / 20.0 * Y1) * 0.6
    Phi_init = -jnp.sin(1.5 * jnp.pi / 20.0 * Y1) * 0.6
    U2_init = Y2
    Psi_init = 0.5 * (1 - jnp.cos(2 * jnp.pi / 20.0 * Y1)) * 0.12

    # Find self-similar solution
    time_start = time.time()
    lambda_, U1, U2, Omega, Phi, Psi = search(
        Y1, Y2, lambda_init, U1_init, U2_init, Phi_init, Psi_init
    )
    print("Time elapsed:", time.time() - time_start)
    print("Found lambda =", lambda_)

    # Plot result
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        1, 5, figsize=(12, 4), subplot_kw={"projection": "3d"}
    )
    ax1.plot_surface(Y1, Y2, U1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_zlim(-20, 20)
    ax1.set_xlabel("y1")
    ax1.set_ylabel("y2")
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(0, 20)
    ax1.set_title("U1")
    ax2.plot_surface(Y1, Y2, U2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_zlim(0, 25)
    ax2.set_xlabel("y1")
    ax2.set_ylabel("y2")
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(0, 20)
    ax2.set_title("U2")
    ax3.plot_surface(Y1, Y2, Omega, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax3.set_zlim(-0.6, 0.6)
    ax3.set_xlabel("y1")
    ax3.set_ylabel("y2")
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(0, 20)
    ax3.set_title("Omega")
    ax4.plot_surface(Y1, Y2, Phi, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax4.set_zlim(-0.6, 0.6)
    ax4.set_xlabel("y1")
    ax4.set_ylabel("y2")
    ax4.set_xlim(-20, 20)
    ax4.set_ylim(0, 20)
    ax4.set_title("Phi")
    ax5.plot_surface(Y1, Y2, Psi, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax5.set_zlim(0, 0.12)
    ax5.set_xlabel("y1")
    ax5.set_ylabel("y2")
    ax5.set_xlim(-20, 20)
    ax5.set_ylim(0, 20)
    ax5.set_title("Psi")

    plt.savefig("sol_boussinesq.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
