import matplotlib.pyplot as plt
import os

def plot_one_iteration_boxplot(avg_mu_dict, avg_std_dict, iter_id, save_dir="iteration_plots", std_scale=2.0):
    """
    Create a boxplot-style visualization for one iteration, where each "box" is based on avg_mu ± scaled avg_std.

    Args:
        avg_mu_dict (dict): region_id -> avg_mu
        avg_std_dict (dict): region_id -> avg_std
        iter_id (int or str): iteration index
        save_dir (str): directory to save output plots
        std_scale (float): scale factor to exaggerate std (e.g., 2.0 makes the boxes taller)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    region_ids = list(avg_mu_dict.keys())
    region_labels = [str(rid) for rid in region_ids]

    # Simulate box data using scaled std
    # box_data = []
    # for rid in region_ids:
    #     mu = avg_mu_dict[rid]
    #     std = avg_std_dict[rid]
    #     q1 = mu - std_scale * std / 2
    #     q3 = mu + std_scale * std / 2
    #     box_data.append([q1, mu, q3])  # pseudo box data
    mu_vals = [avg_mu_dict[rid] for rid in region_ids]
    std_vals = [avg_std_dict[rid] * std_scale / 2 for rid in region_ids]
    x_pos = range(len(region_ids))

    

    # Create the plot
    plt.figure(figsize=(max(10, len(region_ids) * 0.6), 6))
    plt.errorbar(x_pos, mu_vals, yerr=std_vals, fmt='o', ecolor='blue', capsize=4, color='black')
    # plt.boxplot(box_data, showfliers=False, widths=0.6, patch_artist=True,
    #             boxprops=dict(facecolor='lightblue', color='blue'),
    #             medianprops=dict(color='red'))

    plt.xticks(range(1, len(region_labels) + 1), region_labels, rotation=90)
    plt.xlabel("Subregion ID")
    plt.ylabel(f"avg_mu ± {std_scale} × avg_std")
    plt.title(f"Iteration {iter_id}: Boxplot of Subregion GP Summary")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"iter_{iter_id}_boxplot.png")
    plt.savefig(save_path)
    plt.close()