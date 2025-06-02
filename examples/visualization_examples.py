"""
Example usage of visualization functions
Run with: python -m examples.visualization_examples
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from visualization import (
    plot_cluster_metrics,
    show_cluster_visualization,
    show_feature_distributions,
    show_cluster_timeline,
    show_motion_heatmaps,
    create_live_dashboard
)
from config import Config


def generate_example_data():
    """Generate synthetic data for demonstration"""
    np.random.seed(42)

    # Simulate features
    n_samples = 200
    features = pd.DataFrame({
        'mag_mean': np.random.exponential(1, n_samples),
        'mag_std': np.random.normal(1, 0.2, n_samples),
        'mag_max': np.random.exponential(2, n_samples),
        'ang_mean': np.random.uniform(-np.pi, np.pi, n_samples),
        'ang_std': np.random.uniform(0, 1, n_samples),
        'x_flow': np.random.normal(0, 1, n_samples),
        'y_flow': np.random.normal(0, 1, n_samples)
    })

    # Simulate clusters (3 groups)
    cluster_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])

    # Simulate magnitudes and angles for heatmaps
    magnitudes = [np.random.rand(Config.RESIZE_HEIGHT, Config.RESIZE_WIDTH) for _ in range(n_samples)]
    angles = [np.random.uniform(-np.pi, np.pi, (Config.RESIZE_HEIGHT, Config.RESIZE_WIDTH)) for _ in range(n_samples)]

    # Simulate timestamps
    timestamps = np.linspace(0, 10, n_samples)

    return features, cluster_labels, magnitudes, angles, timestamps


def run_visualization_examples():
    """Demonstrate all visualization functions"""
    print("Generating example data...")
    features, cluster_labels, magnitudes, angles, timestamps = generate_example_data()

    print("\n1. Showing cluster metrics visualization...")
    k_range = range(2, 8)
    silhouette_scores = [0.4, 0.5, 0.55, 0.53, 0.51, 0.49]  # Example values
    calinski_scores = [120, 180, 210, 205, 190, 185]  # Example values
    davies_scores = [1.2, 0.9, 0.7, 0.75, 0.8, 0.85]  # Example values

    metrics_fig = plot_cluster_metrics(
        k_range=k_range,
        silhouette_scores=silhouette_scores,
        calinski_scores=calinski_scores,
        davies_scores=davies_scores,
        optimal_k=3
    )
    metrics_fig.show()

    print("\n2. Showing cluster visualization...")
    cluster_fig = show_cluster_visualization(
        features=features.values,
        cluster_labels=cluster_labels
    )
    cluster_fig.show()

    print("\n3. Showing feature distributions...")
    feature_figs = show_feature_distributions(
        features=features,
        cluster_labels=cluster_labels
    )
    for fig in feature_figs:
        fig.show()

    print("\n4. Showing cluster timeline...")
    timeline_fig = show_cluster_timeline(
        timestamps=timestamps,
        cluster_labels=cluster_labels,
        magnitudes=np.array([np.mean(m) for m in magnitudes])
    )
    timeline_fig.show()

    print("\n5. Showing motion heatmaps...")
    heatmap_figs = show_motion_heatmaps(
        magnitudes=magnitudes[:5],  # Just show first 5 for example
        cluster_labels=cluster_labels[:5]
    )
    for fig in heatmap_figs:
        fig.show()

    print("\n6. Showing live dashboard...")
    dashboard = create_live_dashboard(
        frames=[np.zeros((Config.RESIZE_HEIGHT, Config.RESIZE_WIDTH, 3))] * len(timestamps),  # Dummy frames
        magnitudes=magnitudes,
        angles=angles,
        cluster_labels=cluster_labels,
        fps=30
    )
    dashboard.show()


if __name__ == "__main__":
    run_visualization_examples()