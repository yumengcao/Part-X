import matplotlib.pyplot as plt
import numpy as np
import os

def plot_score_comparison_bar(
    scores_unscaled: dict,
    scores_scaled: dict,
    iter_id: int,
    save_dir="score_compare_plots"
):
    """
    Plot bar comparison between unscaled and scaled scores for a given iteration.
    
    Args:
        scores_unscaled (dict): region_id -> score (unscaled)
        scores_scaled (dict): region_id -> score (scaled)
        iter_id (int): iteration number
        save_dir (str): directory to save the plot
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    labels = list(scores_unscaled.keys())
    vals1 = [scores_unscaled[k] for k in labels]
    scale_factor = 100  # scale scaled scores for better visualization
    vals2 = [scores_scaled[k] * scale_factor for k in labels]
    #vals2 = [scores_scaled[k] for k in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 5))
    bars1 = ax.bar(x - width/2, vals1, width, label='Unscaled', color='skyblue')
    bars2 = ax.bar(x + width/2, vals2, width, label='Scaled', color='salmon')

    # Add value labels on top
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Score")
    ax.set_title(f"Iteration {iter_id} - Score Comparison (Scaled x{scale_factor})")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"iter_{iter_id}_score_comparison.png")
    plt.savefig(save_path)
    plt.close()