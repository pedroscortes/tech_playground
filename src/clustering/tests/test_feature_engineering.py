import unittest
import pandas as pd
import numpy as np
from src.clustering.analyzer.feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across test methods"""
        cls.sentiment_results = pd.DataFrame({
            'employee_id': [1, 1, 2, 2, 3],
            'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'confidence': [0.8, 0.7, 0.6, 0.9, 0.75],
            'sentiment_score': [0.6, -0.3, 0.1, 0.8, -0.5],
            'normalized_score': [0.8, 0.35, 0.55, 0.9, 0.25],
            'field': ['feedback', 'career', 'feedback', 'manager', 'career'],
            'original_text': [
                'Great environment',
                'Need more opportunities',
                'Regular day',
                'Good leadership',
                'Limited growth'
            ]
        })
        
        cls.employee_data = pd.DataFrame({
            'id': [1, 2, 3],
            'department_name': ['Tech', 'Sales', 'HR'],
            'position_title': ['Developer', 'Sales Rep', 'HR Analyst'],
            'gender': ['M', 'F', 'M'],
            'generation': ['Gen Y', 'Gen X', 'Gen Y'],
            'company_tenure': ['1-2 years', '2-5 years', '0-1 year']
        })
        
        cls.feature_engineer = FeatureEngineer()
        
    def test_create_sentiment_features(self):
        """Test creation of sentiment-based features"""
        sentiment_features = self.feature_engineer.create_sentiment_features(self.sentiment_results)
        
        self.assertEqual(len(sentiment_features), 3)  
        
        expected_columns = [
            'sentiment_score_feedback',
            'confidence_feedback',
            'normalized_score_feedback'
        ]
        for col in expected_columns:
            self.assertIn(col, sentiment_features.columns)
            
        self.assertGreaterEqual(sentiment_features.values.min(), -1)
        self.assertLessEqual(sentiment_features.values.max(), 1)
    
    def test_create_text_features(self):
        """Test creation of text-based features"""
        text_features = self.feature_engineer.create_text_features(self.sentiment_results)
        
        self.assertEqual(len(text_features), 3)  
        self.assertEqual(len(text_features.columns), 100)  
        
        self.assertTrue(np.issubdtype(text_features.values.dtype, np.number))
        
        self.assertGreaterEqual(text_features.values.min(), 0)
        self.assertLessEqual(text_features.values.max(), 1)
    
    def test_create_temporal_features(self):
        """Test creation of temporal features"""
        temporal_features = self.feature_engineer.create_temporal_features(self.sentiment_results)
        
        required_features = ['sentiment_volatility', 'feedback_frequency', 'avg_confidence']
        for feature in required_features:
            self.assertIn(feature, temporal_features.columns)
        
        self.assertTrue((temporal_features['feedback_frequency'] >= 1).all())
        self.assertTrue((temporal_features['avg_confidence'] >= 0).all())
        self.assertTrue((temporal_features['avg_confidence'] <= 1).all())
    
    def test_create_categorical_features(self):
        """Test creation of categorical features"""
        categorical_features = self.feature_engineer.create_categorical_features(self.employee_data)
        
        self.assertIn('department_Tech', categorical_features.columns)
        self.assertIn('position_Developer', categorical_features.columns)
        self.assertIn('gender_M', categorical_features.columns)
        
        self.assertTrue(set(categorical_features.values.flatten()).issubset({0, 1}))
    
    def test_create_features_integration(self):
        """Test complete feature creation pipeline"""
        features = self.feature_engineer.create_features(
            self.sentiment_results,
            self.employee_data
        )
        
        self.assertTrue(any('sentiment_score' in col for col in features.columns))
        self.assertTrue(any('text_feature' in col for col in features.columns))
        self.assertIn('sentiment_volatility', features.columns)
        self.assertTrue(any('department_' in col for col in features.columns))
        
        self.assertTrue(np.issubdtype(features.values.dtype, np.number))
        
        self.assertFalse(features.isnull().any().any())

if __name__ == '__main__':
    unittest.main(verbosity=2)