import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from src.clustering.analyzer.cluster_analysis import ClusterAnalyzer

class TestClusterAnalyzer(unittest.TestCase):
    """Test cases for ClusterAnalyzer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data using sklearn's make_blobs for clustering"""
        n_samples = 300
        n_features = 10
        n_clusters = 3
        
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            random_state=42
        )
        
        cls.features = pd.DataFrame(
            X,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        cls.true_labels = y
        
        cls.analyzer = ClusterAnalyzer(
            min_clusters=2,
            max_clusters=5,
            random_state=42
        )
    
    def test_scale_features(self):
        """Test feature scaling"""
        scaled_features = self.analyzer.scale_features(self.features)
        
        self.assertEqual(scaled_features.shape, self.features.shape)
        
        self.assertAlmostEqual(scaled_features.mean(), 0, places=1)
        self.assertAlmostEqual(scaled_features.std(), 1, places=1)
    
    def test_evaluate_clusters(self):
        """Test cluster evaluation for a specific k"""
        n_clusters = 3
        scaled_features = self.analyzer.scale_features(self.features)
        
        kmeans, metrics = self.analyzer.evaluate_clusters(
            scaled_features,
            n_clusters
        )
        
        required_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        self.assertGreaterEqual(metrics['silhouette'], -1)
        self.assertLessEqual(metrics['silhouette'], 1)
        self.assertGreater(metrics['calinski_harabasz'], 0)
        self.assertGreater(metrics['davies_bouldin'], 0)
        
        clusters = kmeans.labels_
        self.assertEqual(len(np.unique(clusters)), n_clusters)
    
    def test_find_optimal_clusters(self):
        """Test optimal cluster number selection"""
        scaled_features = self.analyzer.scale_features(self.features)
        
        metrics_history = {}
        for k in range(2, 5):
            _, metrics = self.analyzer.evaluate_clusters(scaled_features, k)
            metrics_history[k] = metrics
        
        optimal_k = self.analyzer.find_optimal_clusters(metrics_history)
        
        self.assertGreaterEqual(optimal_k, 2)
        self.assertLessEqual(optimal_k, 4)
        
        self.assertEqual(optimal_k, 3)
    
    def test_perform_clustering(self):
        """Test complete clustering pipeline"""
        kmeans, clusters, scaled_features, metrics_history = self.analyzer.perform_clustering(
            self.features,
            plot_metrics=False
        )
        
        self.assertEqual(len(clusters), len(self.features))
        self.assertEqual(scaled_features.shape, self.features.shape)
        self.assertGreater(len(metrics_history), 0)
        
        n_clusters = len(np.unique(clusters))
        self.assertGreaterEqual(n_clusters, 2)
        self.assertLessEqual(n_clusters, 5)
        
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(self.true_labels, clusters)
        self.assertGreater(ari, 0.5)  
    
    def test_get_cluster_feature_importance(self):
        """Test feature importance calculation"""
        _, clusters, _, _ = self.analyzer.perform_clustering(
            self.features,
            plot_metrics=False
        )
        
        importance = self.analyzer.get_cluster_feature_importance(
            self.features,
            clusters,
            n_top_features=3
        )
        
        self.assertEqual(len(importance), len(np.unique(clusters)))
        
        for cluster_id, features in importance.items():
            self.assertEqual(len(features), 3)
            
            for feature, score in features:
                self.assertIsInstance(feature, str)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)