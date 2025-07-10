import matplotlib.pyplot as plt
import os

def plot_one_iteration_boxplot(
    avg_mu_dict,
    avg_std_dict,
    iter_id,
    save_dir="iteration_plots",
    std_scale=10.0  # 放大系数：你可以设为 10 或更高
):
    """
    Visualize GP results per subregion in one iteration.
    Plot: center at avg_mu, box height is scaled std.

    Args:
        avg_mu_dict (dict): region_id -> avg_mu
        avg_std_dict (dict): region_id -> avg_std
        iter_id (int or str): iteration number
        save_dir (str): directory to save the plot
        std_scale (float): scale multiplier for std to emphasize box height
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    region_ids = list(avg_mu_dict.keys())
    region_labels = [str(rid) for rid in region_ids]

    box_data = []
    for rid in region_ids:
        mu = avg_mu_dict[rid]
        std = avg_std_dict[rid]
        std_height = std * std_scale
        q1 = mu - std_height / 2
        q3 = mu + std_height / 2
        box_data.append([q1, mu, q3])

    plt.figure(figsize=(max(10, len(region_ids) * 0.6), 6))
    plt.boxplot(
        box_data,
        showfliers=False,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue'),
        medianprops=dict(color='red')
    )

    plt.xticks(range(1, len(region_labels) + 1), region_labels, rotation=90)
    plt.xlabel("Subregion ID")
    plt.ylabel(f"avg_mu ± {std_scale} × avg_std")
    plt.title(f"Iteration {iter_id}: GP Box (std scaled x{std_scale})")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"iter_{iter_id}_boxplot.png")
    plt.savefig(save_path)
    plt.close()
