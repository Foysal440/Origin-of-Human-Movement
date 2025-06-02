import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
import warnings
from config import Config


class ClusteringError(Exception):
    """Custom exception for clustering-related errors"""
    pass


def _validate_input_data(X):
    """Validate input data before clustering"""
    if X is None or len(X) == 0:
        raise ClusteringError("Input data cannot be empty")

    if len(X.shape) != 2:
        raise ClusteringError("Input data must be 2-dimensional")

    if X.shape[0] < 2:
        raise ClusteringError("At least 2 samples required for clustering")


def _safe_cluster_metrics(X, labels, metric_name):
    """Calculate clustering metrics with error handling"""
    try:
        if metric_name == 'silhouette':
            return silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
        elif metric_name == 'calinski':
            return calinski_harabasz_score(X, labels)
        elif metric_name == 'davies':
            return davies_bouldin_score(X, labels)
    except:
        return np.nan


def find_optimal_clusters(X, max_k=10):
    """Determine optimal number of clusters using multiple metrics with robust error handling"""
    try:
        _validate_input_data(X)

        # Adjust max_k based on sample size
        max_k = min(max_k, max(2, X.shape[0] // 5))
        if max_k < 2:
            return 1

        k_range = range(2, max_k + 1)
        scores = {
            'silhouette': [],
            'calinski': [],
            'davies': []
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X)

                scores['silhouette'].append(_safe_cluster_metrics(X, clusters, 'silhouette'))
                scores['calinski'].append(_safe_cluster_metrics(X, clusters, 'calinski'))
                scores['davies'].append(_safe_cluster_metrics(X, clusters, 'davies'))

        # Normalize scores for combined decision
        valid_metrics = [m for m in scores if not all(np.isnan(scores[m]))]
        if not valid_metrics:
            return Config.DEFAULT_CLUSTERS

        combined_scores = np.zeros(len(k_range))
        for metric in valid_metrics:
            metric_scores = np.array(scores[metric])
            if metric == 'davies':  # Lower is better for Davies-Bouldin
                norm_scores = 1 - ((metric_scores - np.nanmin(metric_scores)) /
                                   (np.nanmax(metric_scores) - np.nanmin(metric_scores)))
            else:  # Higher is better
                norm_scores = ((metric_scores - np.nanmin(metric_scores)) /
                               (np.nanmax(metric_scores) - np.nanmin(metric_scores)))
            combined_scores += np.nan_to_num(norm_scores, nan=0)

        best_k = k_range[np.argmax(combined_scores)]
        return max(2, min(best_k, max_k))  # Ensure within bounds

    except Exception as e:
        warnings.warn(f"Optimal cluster detection failed: {str(e)}. Using default clusters.")
        return Config.DEFAULT_CLUSTERS


def perform_clustering(X, algorithm=Config.DEFAULT_ALGORITHM, n_clusters=None):
    """Robust clustering implementation with comprehensive error handling"""
    results = {'metrics': {}}
    scaler = StandardScaler()

    try:
        _validate_input_data(X)

        # Handle potential NaN values
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

        # Scale features
        X_scaled = scaler.fit_transform(X)

        # Determine cluster count if not specified
        if n_clusters is None:
            n_clusters = Config.DEFAULT_CLUSTERS

        # Perform clustering based on algorithm
        if algorithm in ['K-Means', 'Auto-Detect']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)

                k = find_optimal_clusters(X_scaled) if algorithm == 'Auto-Detect' else n_clusters
                k = max(2, min(k, Config.MAX_CLUSTERS))  # Ensure within configured bounds

                kmeans = KMeans(n_clusters=k, random_state=42)
                results['KMeans'] = kmeans.fit_predict(X_scaled)

                # Store metrics
                results['metrics']['KMeans'] = {
                    'silhouette': _safe_cluster_metrics(X_scaled, results['KMeans'], 'silhouette'),
                    'calinski': _safe_cluster_metrics(X_scaled, results['KMeans'], 'calinski'),
                    'davies': _safe_cluster_metrics(X_scaled, results['KMeans'], 'davies'),
                    'n_clusters': k
                }

        if algorithm in ['Bayesian GMM', 'Auto-Detect']:
            max_components = min(Config.MAX_CLUSTERS, X.shape[0] // 5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)

                bgm = BayesianGaussianMixture(
                    n_components=max_components,
                    weight_concentration_prior_type="dirichlet_process",
                    random_state=42,
                    max_iter=200)
                results['BayesianGMM'] = bgm.fit_predict(X_scaled)

                # Store metrics
                results['metrics']['BayesianGMM'] = {
                    'silhouette': _safe_cluster_metrics(X_scaled, results['BayesianGMM'], 'silhouette'),
                    'calinski': _safe_cluster_metrics(X_scaled, results['BayesianGMM'], 'calinski'),
                    'davies': _safe_cluster_metrics(X_scaled, results['BayesianGMM'], 'davies'),
                    'n_components': bgm.n_components_
                }

        return results, scaler

    except Exception as e:
        raise ClusteringError(f"Clustering failed: {str(e)}")


def update_live_model(features, algorithm=Config.DEFAULT_ALGORITHM, n_clusters=None):
    """Incremental clustering update for live data with error handling"""
    try:
        _validate_input_data(features)

        if n_clusters is None:
            n_clusters = Config.DEFAULT_CLUSTERS

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        if algorithm == 'K-Means':
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(100, len(features)),
                random_state=42,
                n_init=3
            ).fit(X_scaled)

        elif algorithm == 'Bayesian GMM':
            max_components = min(Config.MAX_CLUSTERS, len(features) // 5)
            model = BayesianGaussianMixture(
                n_components=max_components,
                weight_concentration_prior_type="dirichlet_process",
                random_state=42,
                max_iter=100
            ).fit(X_scaled)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return model, scaler

    except Exception as e:
        raise ClusteringError(f"Live model update failed: {str(e)}")