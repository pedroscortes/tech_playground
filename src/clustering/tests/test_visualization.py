import unittest
import pandas as pd
import numpy as np
from plotly.graph_objs import Figure
from src.clustering.analyzer.visualization import ClusterVisualizer

class TestClusterVisualizer(unittest.TestCase):
    """Test cases for ClusterVisualizer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for visualizations"""
        np.random.seed(42)
        
        n_samples = 100
        n_features = 10
        cls.features = np.random.randn(n_samples, n_features)
        
        cls.clusters = np.random.randint(0, 3, n_samples)
        
        cls.risk_scores = pd.DataFrame({
            'sentiment_risk': np.random.uniform(0, 1, n_samples),
            'confidence_risk': np.random.uniform(0, 1, n_samples),
            'volatility_risk': np.random.uniform(0, 1, n_samples),
            'engagement_risk': np.random.uniform(0, 1, n_samples),
            'tenure_risk': np.random.uniform(0, 1, n_samples),
            'overall_risk': np.random.uniform(0, 1, n_samples),
            'cluster': cls.clusters
        })
        
        cls.employee_data = pd.DataFrame({
            'department_name': np.random.choice(['Tech', 'Sales', 'HR'], n_samples),
            'generation': np.random.choice(['Gen X', 'Gen Y', 'Gen Z'], n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'company_tenure': np.random.choice(['0-1', '1-2', '2-5', '5+'], n_samples)
        })
        
        cls.importance_by_cluster = {
            i: [
                (f'feature_{j}', np.random.random())
                for j in range(5)
            ]
            for i in range(3)
        }
        
        cls.visualizer = ClusterVisualizer()
    
    def assert_is_valid_figure(self, fig):
        """Helper method to validate Plotly figures"""
        self.assertIsInstance(fig, Figure)
        self.assertTrue(hasattr(fig, 'data'))
        self.assertTrue(hasattr(fig, 'layout'))
    
    def test_plot_cluster_distribution(self):
        """Test cluster distribution plot creation"""
        fig = self.visualizer.plot_cluster_distribution(self.clusters)
        
        self.assert_is_valid_figure(fig)
        
        cluster_counts = pd.Series(self.clusters).value_counts()
        self.assertEqual(len(fig.data[0].x), len(cluster_counts))
        
        self.assertIn('Cluster', fig.layout.xaxis.title.text)
        self.assertIn('Number', fig.layout.yaxis.title.text)
    
    def test_plot_dimension_reduction(self):
        """Test dimension reduction plot creation"""
        fig = self.visualizer.plot_dimension_reduction(
            self.features,
            self.clusters
        )
        
        self.assert_is_valid_figure(fig)
        
        self.assertEqual(len(fig.data[0].x), len(self.clusters))
        self.assertEqual(len(fig.data[0].y), len(self.clusters))
        
        self.assertIn('PC', fig.layout.xaxis.title.text)
        self.assertIn('PC', fig.layout.yaxis.title.text)
    
    def test_plot_feature_importance(self):
        """Test feature importance plot creation"""
        fig = self.visualizer.plot_feature_importance(
            self.importance_by_cluster
        )
        
        self.assert_is_valid_figure(fig)
        
        self.assertEqual(
            len(fig.data[0].x),  
            len(self.importance_by_cluster)
        )
        
        self.assertIn('Cluster', fig.layout.xaxis.title.text)
        self.assertIn('Feature', fig.layout.yaxis.title.text)
    
    def test_plot_risk_profiles(self):
        """Test risk profile plots creation"""
        radar_fig, box_fig = self.visualizer.plot_risk_profiles(
            self.risk_scores,
            self.clusters
        )
        
        self.assert_is_valid_figure(radar_fig)
        self.assertEqual(
            len(radar_fig.data),  
            len(np.unique(self.clusters))
        )
        
        self.assert_is_valid_figure(box_fig)
        self.assertEqual(
            len(box_fig.data),  
            len(np.unique(self.clusters))
        )
    
    def test_plot_demographic_distribution(self):
        """Test demographic distribution plot creation"""
        fig = self.visualizer.plot_demographic_distribution(
            self.employee_data,
            self.clusters
        )
        
        self.assert_is_valid_figure(fig)
        
        demographic_cols = [
            'department_name', 'generation', 'gender', 'company_tenure'
        ]
        self.assertEqual(
            len(fig.data),  
            len(demographic_cols) * len(pd.Series(self.clusters).unique())
        )
    
    def test_create_dashboard(self):
        """Test dashboard creation"""
        try:
            self.visualizer.create_dashboard(
                self.features,
                self.clusters,
                self.risk_scores,
                self.employee_data,
                self.importance_by_cluster
            )
            execution_successful = True
        except Exception as e:
            execution_successful = False
            print(f"Dashboard creation failed: {str(e)}")
        
        self.assertTrue(execution_successful)

if __name__ == '__main__':
    unittest.main(verbosity=2)