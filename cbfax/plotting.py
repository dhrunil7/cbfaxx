import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

def _plot_halfspace_lessthan(normal_vector, constant, xlim=(-10, 10), ylim=(-10, 10), linestyle='-'):
    # Define the normal vector and constant
    a, b = normal_vector
    c = constant

    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)

    # Calculate the values of the halfspace
    Z = a * X + b * Y + c

    # Plot the halfspace
    # plt.contourf(X, Y, Z <= c, alpha=0.5, colors=['#ff9999', '#9999ff'])
    plt.contourf(X, Y, Z <= 0, alpha=0.5, colors=['#ffb09c', '#E0FFD2'])
    plt.contour(X, Y, Z, levels=[0], colors='black', linestyles=linestyle)

    # Set the limits and labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(f'Halfspace: {a}x + {b}y ≤ {c}')

    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    # plt.show()


def plot_halfspace(normal_vector, constant, relation, xlim=(-10, 10), ylim=(-10, 10)):
    if relation == "<=":
        _plot_halfspace_lessthan(normal_vector, constant, xlim=(-10, 10), ylim=(-10, 10), linestyle='-')
    elif relation == "<":
        _plot_halfspace_lessthan(normal_vector, constant, xlim=(-10, 10), ylim=(-10, 10), linestyle='--')
    elif relation == ">=":
        _plot_halfspace_lessthan([-normal_vector[0], -normal_vector[1]], -constant, xlim=(-10, 10), ylim=(-10, 10), linestyle='-')
    elif relation == ">":
        _plot_halfspace_lessthan([-normal_vector[0], -normal_vector[1]], -constant, xlim=(-10, 10), ylim=(-10, 10), linestyle='--')



def interactive_halfspace(a, b, c, relation):
    plot_halfspace([a, b], c, relation)


def plot_cbf(barrier_func, xlim=(-10, 10), ylim=(-10, 10)):
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)

    # Calculate the CBF values
    H = barrier_func(np.stack([X,Y]).reshape(2,-1)).reshape(400, 400)

    # Plot the CBF
    plt.contourf(X, Y, H >= 0, alpha=0.6, colors=['#ff9999', '#99ff99'])
    plt.contourf(X, Y, H, alpha=0.3)
    plt.contour(X, Y, H, levels=[0], colors='black')

    # Set the limits and labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(f'Control Barrier Function: {a}x^2 + {b}y^2 + {c}')

    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

# # Create interactive widgets
# a_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description='a')
# b_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description='b')
# c_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description='c')
# relation_slider = widgets.SelectionSlider(options=["<=", ">=", "<", ">"], description="relation")
# # Use interact to create the sliders
# interact(interactive_halfspace, a=a_slider, b=b_slider, c=c_slider, relation=relation_slider)


# # Example usage
# normal_vector = [1, 1]
# constant = 0
# plot_halfspace(normal_vector, constant)
# # normal_vector = [-1, -1]
# # plot_halfspace(normal_vector, constant)