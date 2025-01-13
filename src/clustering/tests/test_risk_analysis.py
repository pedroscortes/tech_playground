import unittest
import pandas as pd
import numpy as np
from src.clustering.analyzer.risk_analysis import RiskAnalyzer

class TestRiskAnalyzer(unittest.TestCase):
    """Test cases for RiskAnalyzer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for risk analysis"""
        cls.sentiment_results = pd.DataFrame({
            'employee_id': [1, 1, 2, 2, 3, 3],
            'sentiment_label': ['positive', 'negative', 'neutral', 'negative', 'negative', 'negative'],
            'confidence': [0.9, 0.8, 0.7, 0.85, 0.95, 0.9],
            'normalized_score': [0.8, 0.3, 0.5, 0.2, 0.1, 0.15],
            'field': ['feedback', 'career', 'feedback', 'manager', 'career', 'feedback']
        })
        
        cls.features = pd.DataFrame({
            'tenure_0_1_year': [1, 0, 0],
            'tenure_1_2_years': [0, 1, 0],
            'tenure_2_5_years': [0, 0, 1],
            'sentiment_score': [0.2, -0.3, -0.5]
        }, index=[1, 2, 3])
        
        cls.clusters = np.array([0, 1, 1])
        
        cls.risk_analyzer = RiskAnalyzer(
            risk_weights={
                'sentiment_risk': 0.3,
                'confidence_risk': 0.2,
                'volatility_risk': 0.2,
                'engagement_risk': 0.15,
                'tenure_risk': 0.15
            }
        )
    
    def test_calculate_sentiment_risk(self):
        """Test sentiment risk calculation"""
        sentiment_risk = self.risk_analyzer.calculate_sentiment_risk(self.sentiment_results)
        
        self.assertEqual(len(sentiment_risk), 3)
        
        self.assertTrue(all(0 <= score <= 1 for score in sentiment_risk))
        
        self.assertEqual(sentiment_risk[3], 1.0)
        
        self.assertLess(sentiment_risk[1], sentiment_risk[3])
    
    def test_calculate_confidence_risk(self):
        """Test confidence risk calculation"""
        confidence_risk = self.risk_analyzer.calculate_confidence_risk(self.sentiment_results)
        
        self.assertTrue(all(0 <= risk <= 1 for risk in confidence_risk))
        
        expected_risk = 1 - self.sentiment_results.groupby('employee_id')['confidence'].mean()
        pd.testing.assert_series_equal(confidence_risk, expected_risk)
    
    def test_calculate_volatility_risk(self):
        """Test volatility risk calculation"""
        volatility_risk = self.risk_analyzer.calculate_volatility_risk(self.sentiment_results)
        
        self.assertTrue(all(0 <= risk <= 1 for risk in volatility_risk))
        
        scores_std = self.sentiment_results.groupby('employee_id')['normalized_score'].std()
        self.assertTrue(
            volatility_risk[scores_std.idxmax()] > volatility_risk[scores_std.idxmin()]
        )
    
    def test_calculate_engagement_risk(self):
        """Test engagement risk calculation"""
        engagement_risk = self.risk_analyzer.calculate_engagement_risk(self.sentiment_results)
        
        self.assertTrue(all(0 <= risk <= 1 for risk in engagement_risk))
        
        self.assertTrue(all(risk == engagement_risk.iloc[0] for risk in engagement_risk))
    
    def test_calculate_tenure_risk(self):
        """Test tenure risk calculation"""
        tenure_risk = self.risk_analyzer.calculate_tenure_risk(self.features)
        
        self.assertTrue(all(0 <= risk <= 1 for risk in tenure_risk))
        
        self.assertGreater(tenure_risk[1], tenure_risk[3])  
    
    def test_calculate_risk_scores_integration(self):
        """Test complete risk score calculation pipeline"""
        risk_scores, cluster_risks = self.risk_analyzer.calculate_risk_scores(
            self.features,
            self.sentiment_results,
            self.clusters
        )
        
        required_components = [
            'sentiment_risk',
            'confidence_risk',
            'volatility_risk',
            'engagement_risk',
            'tenure_risk',
            'overall_risk'
        ]
        for component in required_components:
            self.assertIn(component, risk_scores.columns)
        
        self.assertTrue(all(0 <= risk <= 1 for risk in risk_scores['overall_risk']))
        
        self.assertEqual(len(cluster_risks), 2) 
        self.assertTrue(all(metric in cluster_risks.columns 
                          for metric in ['mean', 'std', 'min', 'max']))
    
    def test_identify_high_risk_clusters(self):
        """Test high risk cluster identification"""
        risk_scores, cluster_risks = self.risk_analyzer.calculate_risk_scores(
            self.features,
            self.sentiment_results,
            self.clusters
        )
        
        high_risk_clusters = self.risk_analyzer.identify_high_risk_clusters(
            cluster_risks,
            threshold_std=1.0
        )
        
        self.assertIsInstance(high_risk_clusters, list)
        
        self.assertTrue(all(cluster in np.unique(self.clusters) 
                          for cluster in high_risk_clusters))

if __name__ == '__main__':
    unittest.main(verbosity=2)