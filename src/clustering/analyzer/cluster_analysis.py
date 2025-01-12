from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ClusterAnalyzer:
    """
    A class to handle clustering analysis of employee feedback data.
    
    Attributes:
        n_clusters_range (range): Range of cluster numbers to test
        random_state (int): Random state for reproducibility
        scaler (StandardScaler): Scaler for feature standardization
    """
    
    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 11,
        random_state: int = 42
    ):
        """
        Initialize the ClusterAnalyzer.
        
        Args:
            min_clusters (int): Minimum number of clusters to test
            max_clusters (int): Maximum number of clusters to test
            random_state (int): Random state for reproducibility
        """
        self.n_clusters_range = range(min_clusters, max_clusters)
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def scale_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            features: DataFrame of features to scale
            
        Returns:
            Scaled features as numpy array
        """
        return self.scaler.fit_transform(features)
    
    def evaluate_clusters(
        self,
        features: np.ndarray,
        n_clusters: int
    ) -> Tuple[KMeans, Dict[str, float]]:
        """
        Evaluate clustering for a specific number of clusters.
        
        Args:
            features: Scaled features array
            n_clusters: Number of clusters to evaluate
            
        Returns:
            Tuple of (KMeans model, metrics dictionary)
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(features)
        
        metrics = {
            'silhouette': silhouette_score(features, clusters),
            'calinski_harabasz': calinski_harabasz_score(features, clusters),
            'davies_bouldin': davies_bouldin_score(features, clusters),
            'inertia': kmeans.inertia_
        }
        
        return kmeans, metrics
    
    def plot_validation_metrics(
        self,
        metrics_history: Dict[int, Dict[str, float]]
    ) -> None:
        """
        Plot clustering validation metrics.
        
        Args:
            metrics_history: Dictionary of metrics for each cluster number
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'Silhouette Score',
                'Calinski-Harabasz Score',
                'Davies-Bouldin Score',
                'Inertia (Elbow Method)'
            ]
        )
        
        metric_positions = {
            'silhouette': (1, 1),
            'calinski_harabasz': (1, 2),
            'davies_bouldin': (2, 1),
            'inertia': (2, 2)
        }
        
        for metric, (row, col) in metric_positions.items():
            scores = [metrics[metric] for metrics in metrics_history.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=list(metrics_history.keys()),
                    y=scores,
                    mode='lines+markers',
                    name=metric
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            height=800,
            title_text="Clustering Validation Metrics",
            showlegend=False
        )
        fig.show()
    
    def find_optimal_clusters(
        self,
        metrics_history: Dict[int, Dict[str, float]]
    ) -> int:
        """
        Find optimal number of clusters based on metrics.
        
        Args:
            metrics_history: Dictionary of metrics for each cluster number
            
        Returns:
            Optimal number of clusters
        """
        silhouette_scores = {
            k: v['silhouette'] for k, v in metrics_history.items()
        }
        return max(silhouette_scores.items(), key=lambda x: x[1])[0]
    
    def perform_clustering(
        self,
        features: pd.DataFrame,
        plot_metrics: bool = True
    ) -> Tuple[KMeans, np.ndarray, np.ndarray, Dict[int, Dict[str, float]]]:
        """
        Perform complete clustering analysis.
        
        Args:
            features: DataFrame of features for clustering
            plot_metrics: Whether to plot validation metrics
            
        Returns:
            Tuple of (KMeans model, clusters, scaled features, metrics history)
        """
        try:
            print("\nPerforming clustering analysis...")
            
            scaled_features = self.scale_features(features)
            
            metrics_history = {}
            for k in self.n_clusters_range:
                print(f"Testing {k} clusters...")
                kmeans, metrics = self.evaluate_clusters(scaled_features, k)
                metrics_history[k] = metrics
            
            if plot_metrics:
                self.plot_validation_metrics(metrics_history)
            
            optimal_k = self.find_optimal_clusters(metrics_history)
            print(f"\nOptimal number of clusters: {optimal_k}")
            
            final_kmeans, _ = self.evaluate_clusters(scaled_features, optimal_k)
            final_clusters = final_kmeans.fit_predict(scaled_features)
            
            cluster_dist = pd.Series(final_clusters).value_counts().sort_index()
            print("\nCluster distribution:")
            for cluster_id, size in cluster_dist.items():
                print(f"Cluster {cluster_id}: {size} employees")
            
            return final_kmeans, final_clusters, scaled_features, metrics_history
            
        except Exception as e:
            print(f"Error in clustering analysis: {str(e)}")
            raise

    def get_cluster_feature_importance(
        self,
        features: pd.DataFrame,
        clusters: np.ndarray,
        n_top_features: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Calculate feature importance for each cluster.
        
        Args:
            features: Original features DataFrame
            clusters: Cluster assignments
            n_top_features: Number of top features to return per cluster
            
        Returns:
            Dictionary mapping cluster IDs to lists of (feature, importance) tuples
        """
        importance_by_cluster = {}
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_mean = features[cluster_mask].mean()
            overall_mean = features.mean()
            
            importance = abs(cluster_mean - overall_mean)
            importance = importance / importance.max()
            
            top_features = sorted(
                zip(features.columns, importance),
                key=lambda x: x[1],
                reverse=True
            )[:n_top_features]
            
            importance_by_cluster[cluster_id] = top_features
        
        return importance_by_cluster