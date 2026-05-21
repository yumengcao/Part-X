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
    
    step = 0.05
    xx = np.arange(region[0][0], region[0][1] + 1e-12, step)
    yy = np.arange(region[1][0], region[1][1] + 1e-12, step)
    X, Y = np.meshgrid(xx, yy)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N,2)

    # 试图用向量化接口先评估一次，若失败再逐点评估（兼容多种 test_function 签名）
    
    try:
        # 有些实现返回 shape (N,) 或 (N,1)
        Zvals = np.asarray(test_function(pts))
        Zvals = Zvals.ravel()
        if Zvals.size != pts.shape[0]:
            raise ValueError("vectorized returned wrong size")
    except Exception:
        # 逐点回退：试 func(x, r) -> list 或 func(x)
        Zvals = np.empty(pts.shape[0], dtype=float)
        for i in range(pts.shape[0]):
            x = pts[i]
            try:
                out = test_function(x, 1)   # 先尝试带 replication
            except TypeError:
                out = test_function(x)     # 再尝试不带 replication
            # 如果返回数组/列表，取均值，否则直接标量
            if hasattr(out, "__iter__"):
                Zvals[i] = float(np.mean(out))
            else:
                Zvals[i] = float(out)

    # reshape回网格
    Z = Zvals.reshape(X.shape)

    # 确保补丁（subregion rectangles）画在较低 zorder（比如 zorder=1）
    # 然后把 contour 画在上层
    ax.contour(X, Y, Z, levels=[0], colors='k', linewidths=1.2, zorder=100)
    #--- plot all samples
    for iter_name in sample_all:
        for region_name in sample_all[iter_name]:
            x = sample_all[iter_name][region_name]
            y = rob_all[iter_name][region_name]
            x_pos = x[y > 0]
            x_neg = x[y <= 0]
            if len(x_pos) > 0:
                ax.scatter(x_pos[:, 0], x_pos[:, 1], c='blue', s=2, alpha=0.3)
            if len(x_neg) > 0:
                ax.scatter(x_neg[:, 0], x_neg[:, 1], c='red', s=2, alpha=0.3)

    #ax.set_title(method + ' (partition + samples)'+f' - Iteration {iter_id}', fontsize=10)
    #ax.legend(fontsize='x-small', loc='upper right')
    save_path = os.path.join(save_dir, f"iter_{iter_id}_partitioning_plot.png")
    plt.savefig(save_path)
    plt.close()