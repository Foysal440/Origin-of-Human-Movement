import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Union
from config import Config


class VisualizationError(Exception):
    """Custom exception for visualization errors"""
    pass


def _validate_visualization_input(features: Union[np.ndarray, pd.DataFrame],
                                  labels: np.ndarray) -> None:
    """Validate inputs for visualization functions"""
    if len(features) != len(labels):
        raise VisualizationError("Features and labels must have same length")
    if len(features) == 0:
        raise VisualizationError("Input data cannot be empty")


def plot_cluster_metrics(k_range: range,
                         silhouette_scores: List[float],
                         calinski_scores: List[float],
                         davies_scores: List[float],
                         optimal_k: int) -> go.Figure:
    """Plot cluster evaluation metrics"""
    try:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=(
                                'Silhouette Score (higher → better)',
                                'Calinski-Harabasz (higher → better)',
                                'Davies-Bouldin (lower → better)'
                            ),
                            horizontal_spacing=0.1)

        # Silhouette plot
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=silhouette_scores,
                mode='lines+markers+text',
                name='Silhouette',
                text=[f"{s:.2f}" for s in silhouette_scores],
                textposition="top center",
                line=dict(width=2, color=Config.CLUSTER_COLORS[0])
            ),
            row=1, col=1
        )

        # Calinski plot
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=calinski_scores,
                mode='lines+markers+text',
                name='Calinski',
                text=[f"{s:.0f}" for s in calinski_scores],
                textposition="top center",
                line=dict(width=2, color=Config.CLUSTER_COLORS[1])
            ),
            row=1, col=2
        )

        # Davies plot
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=davies_scores,
                mode='lines+markers+text',
                name='Davies',
                text=[f"{s:.2f}" for s in davies_scores],
                textposition="bottom center",
                line=dict(width=2, color=Config.CLUSTER_COLORS[2])
            ),
            row=1, col=3
        )

        # Add optimal k indicators
        for col in [1, 2, 3]:
            fig.add_vline(
                x=optimal_k,
                line=dict(width=2, dash="dot", color="red"),
                annotation_text=f"Optimal k={optimal_k}",
                annotation_position="top right",
                row=1, col=col
            )

        fig.update_layout(
            title=dict(text='<b>Cluster Quality Metrics</b>', x=0.5, font=dict(size=18)),
            showlegend=False,
            height=450,
            margin=dict(l=40, r=40, b=80, t=100),
            hovermode="x unified",
            plot_bgcolor='rgba(240,240,240,0.8)'
        )

        fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=1)
        fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=2)
        fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=3)

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create metrics plot: {str(e)}")


def show_cluster_visualization(features: Union[np.ndarray, pd.DataFrame],
                               cluster_labels: np.ndarray,
                               frame_numbers: Optional[np.ndarray] = None) -> go.Figure:
    """Show cluster visualization in reduced dimension space"""
    try:
        _validate_visualization_input(features, cluster_labels)

        if frame_numbers is None:
            frame_numbers = np.arange(len(features))

        # Dimensionality reduction
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(features)

        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(features)

        # Create figure with multiple views
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(
                f'PCA 3D Projection (Variance Explained: {100 * pca.explained_variance_ratio_.sum():.1f}%)',
                't-SNE 3D Projection'
            ),
            horizontal_spacing=0.05
        )

        # Create color scale based on cluster colors
        unique_clusters = np.unique(cluster_labels)
        colors = [Config.CLUSTER_COLORS[i % len(Config.CLUSTER_COLORS)] for i in unique_clusters]
        colors = [f'rgb({c[0]},{c[1]},{c[2]})' for c in colors]

        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id

            # PCA plot
            fig.add_trace(
                go.Scatter3d(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    z=X_pca[mask, 2],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(
                        size=5,
                        color=colors[i],
                        opacity=0.8,
                        line=dict(width=0.5, color='DarkSlateGrey')
                    )
                ),
                row=1, col=1
            )

            # t-SNE plot
            fig.add_trace(
                go.Scatter3d(
                    x=X_tsne[mask, 0],
                    y=X_tsne[mask, 1],
                    z=X_tsne[mask, 2],
                    mode='markers',
                    showlegend=False,
                    marker=dict(
                        size=5,
                        color=colors[i],
                        opacity=0.8,
                        line=dict(width=0.5, color='DarkSlateGrey')
                    )
                ),
                row=1, col=2
            )

        fig.update_layout(
            title=dict(text='<b>Cluster Visualization in 3D Space</b>', x=0.5, font=dict(size=18)),
            height=600,
            margin=dict(l=0, r=0, b=80, t=100)
        )

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create cluster visualization: {str(e)}")


def show_feature_distributions(features: pd.DataFrame,
                               cluster_labels: np.ndarray,
                               features_to_show: Optional[List[str]] = None) -> go.Figure:
    """Show distributions of key features by cluster"""
    try:
        _validate_visualization_input(features, cluster_labels)

        if features_to_show is None:
            features_to_show = ['mag_mean', 'mag_std', 'mag_max', 'ang_mean', 'ang_std']

        # Select only numeric features that exist in the DataFrame
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        features_to_show = [f for f in features_to_show if f in numeric_features]

        if not features_to_show:
            raise VisualizationError("No valid numeric features to display")

        # Create combined DataFrame for plotting
        plot_data = features[features_to_show].copy()
        plot_data['Cluster'] = cluster_labels.astype(str)

        fig = px.parallel_coordinates(
            plot_data,
            color="Cluster",
            dimensions=features_to_show,
            title="Parallel Coordinates Plot of Feature Distributions"
        )

        fig.update_layout(
            height=500,
            margin=dict(l=80, r=80, b=80, t=100)
        )

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create feature distributions: {str(e)}")


def show_cluster_timeline(timestamps: np.ndarray,
                          cluster_labels: np.ndarray,
                          magnitudes: np.ndarray) -> go.Figure:
    """Show cluster assignments over time"""
    try:
        _validate_visualization_input(timestamps, cluster_labels)

        # Normalize magnitudes for visualization
        sizes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
        sizes = 10 + 20 * sizes  # Scale to reasonable marker sizes

        fig = px.scatter(
            x=timestamps,
            y=cluster_labels,
            color=cluster_labels.astype(str),
            size=sizes,
            title='Cluster Timeline',
            labels={'x': 'Time (s)', 'y': 'Cluster ID'}
        )

        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.7
            )
        )

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create timeline: {str(e)}")


def show_motion_heatmaps(magnitudes: List[np.ndarray],
                         cluster_labels: np.ndarray,
                         original_frame: Optional[np.ndarray] = None) -> List[go.Figure]:
    """Show heatmaps of motion patterns"""
    try:
        _validate_visualization_input(magnitudes, cluster_labels)

        figs = []
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            if not mask.any():
                continue

            avg_magnitude = np.mean(np.array(magnitudes)[mask], axis=0)
            fig = px.imshow(
                avg_magnitude,
                color_continuous_scale='Jet',
                title=f'Cluster {cluster_id} Motion Pattern'
            )
            figs.append(fig)

        return figs

    except Exception as e:
        raise VisualizationError(f"Failed to create heatmaps: {str(e)}")


def create_live_dashboard(frames: List[np.ndarray],
                          magnitudes: List[np.ndarray],
                          angles: List[np.ndarray],
                          cluster_labels: np.ndarray,
                          fps: float) -> go.Figure:
    """Create comprehensive live analysis dashboard"""
    try:
        n_frames = len(frames)
        if n_frames == 0:
            raise VisualizationError("No frames provided")

        # Create subplots with custom layout
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{"type": "image"}, {"type": "xy"}, {"type": "polar"}],
                [{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=(
                "Current Frame with Motion",
                "Cluster Timeline",
                "Direction Distribution",
                "3D Feature Space",
                "Motion Intensity",
                "Cluster Duration"
            ),
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )

        # Current frame with motion overlay
        current_frame = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
        fig.add_trace(go.Image(z=current_frame), row=1, col=1)
        fig.add_trace(
            go.Heatmap(
                z=magnitudes[-1],
                colorscale='Jet',
                opacity=0.6,
                showscale=False
            ),
            row=1, col=1
        )

        # Cluster timeline
        timestamps = np.arange(n_frames) / fps
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=timestamps[mask],
                    y=cluster_labels[mask],
                    mode='markers',
                    marker=dict(
                        color=f'rgb{Config.CLUSTER_COLORS[cluster_id % len(Config.CLUSTER_COLORS)]}',
                        size=8
                    ),
                    name=f'Cluster {cluster_id}',
                    showlegend=False
                ),
                row=1, col=2
            )

        # Direction distribution
        fig.add_trace(
            go.Histogrampolar(
                r=np.histogram(np.rad2deg(np.concatenate(angles)), bins=36)[0],
                theta=np.linspace(0, 360, 36),
                marker_color='royalblue'
            ),
            row=1, col=3
        )

        # Update polar plot
        fig.update_polars(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickvals=list(range(0, 360, 45))
            ),
            row=1, col=3
        )

        # 3D feature space
        fig.add_trace(
            go.Scatter3d(
                x=np.random.randn(n_frames),
                y=np.random.randn(n_frames),
                z=np.random.randn(n_frames),
                mode='markers',
                marker=dict(
                    size=5,
                    color=cluster_labels,
                    colorscale=[f'rgb{c}' for c in Config.CLUSTER_COLORS]
                )
            ),
            row=2, col=1
        )

        # Motion intensity over time
        avg_magnitudes = [np.mean(m) for m in magnitudes]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=avg_magnitudes,
                mode='lines',
                line=dict(color='firebrick', width=2)
            ),
            row=2, col=2
        )

        # Cluster duration histogram
        cluster_durations = []
        for cluster_id in np.unique(cluster_labels):
            cluster_times = timestamps[cluster_labels == cluster_id]
            if len(cluster_times) > 0:
                duration = np.max(cluster_times) - np.min(cluster_times)
                cluster_durations.append(duration)

        fig.add_trace(
            go.Histogram(
                x=cluster_durations,
                marker_color='forestgreen'
            ),
            row=2, col=3
        )

        # Update layout
        fig.update_layout(
            title_text=f"<b>Human Movement Analysis Dashboard</b><br>Frames: {n_frames} | FPS: {fps:.1f}",
            height=900,
            margin=dict(l=60, r=60, b=100, t=120),
            showlegend=False,
            template="plotly_white"
        )

        return fig

    except Exception as e:
        raise VisualizationError(f"Failed to create dashboard: {str(e)}")