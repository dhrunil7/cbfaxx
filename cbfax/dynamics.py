import jax
import jax.numpy as jnp
import equinox as eqx
import functools
from typing import Callable

def runge_kutta_integrator(dynamics, dt=0.01):
    # zero-order hold
    def integrator(x, u, t):
        dt2 = dt / 2.0
        k1 = dynamics(x, u, t)
        k2 = dynamics(x + dt2 * k1, u, t + dt2)
        k3 = dynamics(x + dt2 * k2, u, t + dt2)
        k4 = dynamics(x + dt * k3, u, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return integrator

@functools.partial(jax.jit, static_argnames=["dynamics"])
def linearize(dynamics, state, control, t):
    A, B = jax.jacobian(dynamics, [0, 1])(state, control, t)
    C = dynamics(state, control, t) - A @ state - B @ control
    return A, B, C

class Dynamics(eqx.Module):
    state_dim: int
    control_dim: int

    def ode_dynamics(self, state, control, t):
        pass  # implement later

    def discrete_step(self, state, control, t, dt):
        return runge_kutta_integrator(self.ode_dynamics, dt)(state, control, t)


class ControlAffineDynamics(Dynamics):
    drift_term: Callable
    control_jacobian: Callable

    def ode_dynamics(self, state, control, t):
        return self.drift_term(state, t) + self.control_jacobian(state, t) @ control


class IntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        self.state_dim = self.integrator_dim * self.N_dim
        self.control_dim = self.N_dim

        A = jnp.eye(self.state_dim, k=self.N_dim)
        B = jnp.zeros([self.state_dim, self.control_dim])
        B = B.at[-self.N_dim:].set(jnp.eye(self.N_dim))
        self.drift_term = lambda x,t: A @ x
        self.control_jacobian = lambda x,t: B
        
    # def ode_dynamics(self, state, control, t):
    #     return jnp.concatenate([state[self.N_dim:], control])
    
def DoubleIntegrator2D():
    return IntegratorND(2, 2) 

def DoubleIntegrator1D():
    return IntegratorND(2, 1) 

def SingleIntegrator2D():
    return IntegratorND(1, 2) 

def SingleIntegrator1D():
    return IntegratorND(1, 1) 

class Unicycle(Dynamics):
    state_dim: int = 3
    control_dim: int = 2

    def ode_dynamics(self, state, control, t):
        x, y, th = state
        v, om = control
        return jnp.array([v * jnp.cos(th),
                          v * jnp.sin(th),
                          om])
    
class DynamicallyExtendedUnicycle(Dynamics):
    state_dim: int = 4
    control_dim: int = 2

    def ode_dynamics(self, state, control, t):
        x, y, th, v = state
        a, om = control
        return jnp.array([v * jnp.cos(th),
                          v * jnp.sin(th),
                          om,
                          a])
    
class SimpleCar(Dynamics):
    state_dim: int = 3
    control_dim: int = 2
    length: int = 1.

    def ode_dynamics(self, state, control, t):
        x, y, th = state
        v, delta = control
        return jnp.array([v * jnp.cos(th),
                          v * jnp.sin(th),
                          v / self.length * jnp.tan(delta)])
    
class DynamicallyExtendedSimpleCar(Dynamics):
    state_dim: int = 4
    control_dim: int = 2
    length: int = 1.

    def ode_dynamics(self, state, control, t):
        x, y, th, v = state
        a, delta = control
        return jnp.array([v * jnp.cos(th),
                          v * jnp.sin(th),
                          v / self.length * jnp.tan(delta),
                          a])
    

class TwoPlayerRelativeIntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        self.state_dim = self.integrator_dim * self.N_dim
        self.control_dim = self.N_dim * 2

        A = jnp.eye(self.state_dim, k=self.N_dim)
        B = jnp.zeros([self.state_dim, self.N_dim])
        B = B.at[-self.N_dim:].set(jnp.eye(self.N_dim))
        B2 = jnp.concatenate([-B, B], axis=-1)
        self.drift_term = lambda x,t: A @ x
        self.control_jacobian = lambda x,t: B2


class RelativeUnicycle(Dynamics):
    state_dim: int = 3
    control_dim: int = 4

    def ode_dynamics(self, state, control, t):
        xrel, yrel, threl = state
        v1, om1, v2, om2 = control
        return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
                          v2 * jnp.sin(threl) - om1 * xrel,
                          om2 - om1])

class RelativeDynamicallyExtendedUnicycle(Dynamics):
    state_dim: int = 5
    control_dim: int = 4

    def ode_dynamics(self, state, control, t):
        xrel, yrel, threl, v1, v2 = state
        a1, om1, a2, om2 = control
        return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
                          v2 * jnp.sin(threl) - om1 * xrel,
                          om2 - om1,
                          a1,
                          a2])