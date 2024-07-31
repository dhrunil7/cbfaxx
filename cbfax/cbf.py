import jax
import jax.numpy as jnp
import functools
from cbfax.dynamics import ControlAffineDynamics, Dynamics, linearize

def lie_derivative(directional_func, scalar_func):
    '''
     ∇b(x)ᵀf(x,t)  b: scalar_func, f: directional_func
    '''
    return lambda state: jax.jacobian(scalar_func)(state) @ directional_func(state)

def lie_derivative_n(order, directional_func, scalar_func):
    '''
     Lⁿfb  b: scalar_func, f: directional_func. Applying it n times.
    '''
    sf = scalar_func
    for n in range(order):
        sf = lie_derivative(directional_func, sf)
    return sf


@functools.partial(jax.jit, static_argnames=["b", "alpha", "dynamics"])
def get_ccbf_constraint_rd1(b, alpha, dynamics, state, control, t):
    if isinstance(dynamics, ControlAffineDynamics):
        drift = lambda x: dynamics.drift_term(x, t)
        control_jac = lambda x: dynamics.control_jacobian(x, t)
    elif isinstance(dynamics, Dynamics):
        A, B, C = linearize(dynamics.ode_dynamics, state, control, t)
        drift = lambda x: A @ state + C
        control_jac = lambda x: B
    else:
        raise ValueError("Invalid dynamics")

    constant = lie_derivative(drift, b)(state) + alpha(b(state))
    linear = lie_derivative(control_jac, b)(state)
    return linear, constant

