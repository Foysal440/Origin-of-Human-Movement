import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from PyQt6.QtCore import QThread, pyqtSignal


class OpticalFlowAnalyzerWorker(QThread):
    analysis_finished = pyqtSignal(dict, np.ndarray)

    def __init__(self, optical_flow_data, method, eps, min_samples):
        super().__init__()
        self.optical_flow_data = optical_flow_data
        self.method = method
        self.eps = eps
        self.min_samples = min_samples

    def run(self):
        try:
            all_flow_data = np.vstack(
                [frame['flow_points'] for frame in self.optical_flow_data if 'flow_points' in frame])

            if len(all_flow_data) > 5000:
                indices = np.random.choice(len(all_flow_data), size=5000, replace=False)
                sampled_flow_data = all_flow_data[indices]
            else:
                sampled_flow_data = all_flow_data

            scaler = StandardScaler()
            X = scaler.fit_transform(sampled_flow_data)

            algorithms = {
                'K-Means': self.run_kmeans,
                'DBSCAN': self.run_dbscan,
                'Hierarchical': self.run_hierarchical,
                'OPTICS': self.run_optics
            }

            clustering_results = {}
            for name, algorithm in algorithms.items():
                try:
                    clustering_results[name] = algorithm(X)
                except Exception as e:
                    print(f"Error in {name}: {str(e)}")
                    clustering_results[name] = np.zeros(len(X))

            self.analysis_finished.emit(clustering_results, X)

        except Exception as e:
            print(f"[Worker] Error: {str(e)}")

    def run_kmeans(self, X):
        max_k = min(10, len(X) - 1)
        best_k = 2
        best_score = -1

        if max_k < 2:
            return np.zeros(len(X))

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def run_dbscan(self, X):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return dbscan.fit_predict(X)

    def run_hierarchical(self, X):
        max_k = min(10, len(X) - 1)
        best_k = 2
        best_score = -1

        if max_k < 2:
            return np.zeros(len(X))

        for k in range(2, max_k + 1):
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        agg = AgglomerativeClustering(n_clusters=best_k)
        return agg.fit_predict(X)

    def run_optics(self, X):
        optics = OPTICS(min_samples=self.min_samples, xi=0.05)
        return optics.fit_predict(X)