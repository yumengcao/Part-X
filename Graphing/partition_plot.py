import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def part_plot(theta_minus, theta_plus, theta_undefined, region, test_function, 
              method, sample_all, rob_all, iter_id, save_dir="partition_plots"):
    fig, ax = plt.subplots(figsize=(8, 8)) 
    ax.set_xlim(region[0][0], region[0][1]) 
    ax.set_ylim(region[1][0], region[1][1])
    seen_regions = set()
    # --- plot theta_minus
    for level in theta_minus:
        for bounds in theta_minus[level].values():
            region_id = tuple(map(tuple, bounds))  
            if region_id in seen_regions:
                continue
            seen_regions.add(region_id)
            ax.add_patch(patches.Rectangle(
                (bounds[0][0], bounds[1][0]),
                bounds[0][1] - bounds[0][0],
                bounds[1][1] - bounds[1][0],
                alpha=0.2, facecolor='r', edgecolor='black'
            ))

    # --- plot theta_plus
    for level in theta_plus:
        for bounds in theta_plus[level].values():
            region_id = tuple(map(tuple, bounds))  
            if region_id in seen_regions:
                continue
            seen_regions.add(region_id)
            ax.add_patch(patches.Rectangle(
                (bounds[0][0], bounds[1][0]),
                bounds[0][1] - bounds[0][0],
                bounds[1][1] - bounds[1][0],
                alpha=0.12, facecolor='g', edgecolor='black'
            ))

    # --- plot theta_undefined
    for bounds in theta_undefined.values():
        ax.add_patch(patches.Rectangle(
            (bounds[0][0], bounds[1][0]),
            bounds[0][1] - bounds[0][0],
            bounds[1][1] - bounds[1][0],
            alpha=0.2, facecolor='b', edgecolor='black'
        ))

    # --- plot contour f(x) = 0
    xx = np.arange(region[0][0], region[0][1], 0.05)
    yy = np.arange(region[1][0], region[1][1], 0.05)
    X, Y = np.meshgrid(xx, yy)
    f_str = test_function.replace('X[0]', 'X').replace('X[1]', 'Y')
    Z = eval(f_str)
    ax.contour(X, Y, Z, levels=[0], colors='k')

    # --- plot all samples
    for iter_name in sample_all:
        for region_name in sample_all[iter_name]:
            x = sample_all[iter_name][region_name]
            y = rob_all[iter_name][region_name]
            x_pos = x[y > 0]
            x_neg = x[y <= 0]
            if len(x_pos) > 0:
                ax.scatter(x_pos[:, 0], x_pos[:, 1], c='blue', s=2)
            if len(x_neg) > 0:
                ax.scatter(x_neg[:, 0], x_neg[:, 1], c='red', s=2)

    ax.set_title(method + ' (partition + samples)')
    ax.legend(fontsize='x-small', loc='upper right')
    save_path = os.path.join(save_dir, f"iter_{iter_id}_partitioning_plot.png")
    plt.savefig(save_path)
    plt.close()