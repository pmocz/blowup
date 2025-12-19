import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

"""
Search for Self-Similar solution in the Burgers 1D equation

Philip Mocz (2025)

Start with the 1D inviscid Burgers (periodic) equation:
u_t + u u_x = 0

Make the ansatz:
u(x, t) = (1-t)^lambda * U(y), where y = x / (1-t)^(-1-lambda)

Consider odd solutions, solutions to:
y + U + U^(1+1/lambda) = 0

lambda = 1/2, 1/4, 1/6, ...
U(-2) = 1
"""


def U_analytic(y):
    """Analytic solution for lambda=1/2 case."""
    A = -9 * y + jnp.sqrt(3) * jnp.sqrt(4 + 27 * y * y)
    term1 = -((2 / 3) ** (1 / 3)) / (A ** (1 / 3))
    term2 = (A ** (1 / 3)) / (2 ** (1 / 3) * 3 ** (2 / 3))
    return term1 + term2


def residual(U, y, lambda_):
    """Compute the residual of the self-similar Burgers equation."""
    return jnp.linalg.norm(y + U + U ** (1 + 1 / lambda_))


def search(y, lambda_, U_init, num_iters=1000, lr=0.001):
    """Find the self-similar solution U(y) using Adam optimizer."""
    U = U_init
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(U)

    for _ in range(num_iters):
        res_grad = jax.grad(residual)(U, y, lambda_)
        updates, opt_state = optimizer.update(res_grad, opt_state)
        U = optax.apply_updates(U, updates)
    return U


def main():
    """Search and plot results."""
    # Parameters
    lambda_ = 1 / 2
    y = jnp.linspace(-2, 2, 400)
    # Initial guess
    U_init = -jnp.tanh(y) / jnp.tanh(2)

    # Find self-similar solution
    U_solution = search(y, lambda_, U_init)

    # Plot result
    plt.plot(y, U_init, label="initial guess", linestyle="dotted", color="gray")
    plt.plot(
        y,
        U_analytic(y),
        label="analytic solution",
        linestyle="dashed",
        color="black",
        linewidth=2,
    )
    plt.plot(y, U_solution, color="red", label="found solution")
    plt.title(f"Self-Similar Solution for λ={lambda_}")
    plt.xlabel("y")
    plt.ylabel("U(y)")
    plt.xlim(-2, 2)
    plt.legend()
    plt.savefig("sol_burgers.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
