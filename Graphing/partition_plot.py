import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from Graphing.sampling_plot import sample_plot


def part_plot(theta_minus, theta_plus, theta_undefined, region, test_function, method, sample_all, rob, group):
    fig, ax = plt.subplots(figsize=(8, 8)) 
    plt.title(method + '_'+ "subregions")
    plt.xlim(region[0][0], region[0][1]) 
    plt.ylim(region[1][0], region[1][1])

    # --- plot theta_minus
    for level in theta_minus:
        for bounds in theta_minus[level].values():
            ax.add_patch(
                patches.Rectangle(
                    (bounds[0][0], bounds[1][0]),
                    bounds[0][1] - bounds[0][0],
                    bounds[1][1] - bounds[1][0],
                    alpha=0.05, facecolor='r', edgecolor='black'
                )
            )

    # --- plot theta_plus
    for level in theta_plus:
        for bounds in theta_plus[level].values():
            ax.add_patch(
                patches.Rectangle(
                    (bounds[0][0], bounds[1][0]),
                    bounds[0][1] - bounds[0][0],
                    bounds[1][1] - bounds[1][0],
                    alpha=0.05, facecolor='g', edgecolor='black'
                )
            )

    # --- plot theta_undefined
    for bounds in theta_undefined.values():
        ax.add_patch(
            patches.Rectangle(
                (bounds[0][0], bounds[1][0]),
                bounds[0][1] - bounds[0][0],
                bounds[1][1] - bounds[1][0],
                alpha=0.05, facecolor='b', edgecolor='black'
            )
        )

    # --- plot contour line for f(x)=0
    xx = np.arange(region[0][0], region[0][1], 0.05)
    yy = np.arange(region[1][0], region[1][1], 0.05)
    X, Y = np.meshgrid(xx, yy)
    f_str = test_function.replace('X[0]', 'X').replace('X[1]', 'Y')
    Z = eval(f_str)
    ax.contour(X, Y, Z, levels=[0], colors='k')

    # --- plot samples on same ax
    sample_plot(ax, sample_all, rob, method, group)

    ax.set_title(method + '_' + group + ' (partition + samples)')
    ax.legend()
    plt.show()