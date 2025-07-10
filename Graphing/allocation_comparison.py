import numpy as np

def plot_allocation_bar(
    alloc1: dict, alloc2: dict, iter_id: int, save_dir="allocation_plots"
):
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    labels = list(alloc1.keys())
    val1 = [alloc1[k] for k in labels]
    val2 = [alloc2[k] for k in labels]
    x = np.arange(len(labels))

    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 5))
    bars1 = ax.bar(x - width / 2, val1, width, label='Unscaled', color='skyblue')
    bars2 = ax.bar(x + width / 2, val2, width, label='Scaled', color='salmon')

    # 添加顶端标注
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Samples Allocated")
    ax.set_title(f"Iteration {iter_id} - Allocation Comparison")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig.savefig(os.path.join(save_dir, f"iter_{iter_id}_allocation_compare.png"))
    plt.close()