import unittest
import pandas as pd
import numpy as np
from src.clustering.analyzer.personas import PersonaGenerator

class TestPersonaGenerator(unittest.TestCase):
    """Test cases for PersonaGenerator class"""
        
    @classmethod
    def setUpClass(cls):
        """Set up test data for persona generation"""
        cls.sentiment_results = pd.DataFrame({
            'employee_id': [1, 1, 2, 2, 3],
            'sentiment_label': ['positive', 'negative', 'positive', 'positive', 'negative'],
            'confidence': [0.9, 0.8, 0.85, 0.9, 0.75],
            'sentiment_score': [0.6, -0.3, 0.5, 0.7, -0.5],
            'normalized_score': [0.8, 0.35, 0.7, 0.85, 0.25],
            'original_text': [
                'Great team environment',  
                'Need better career path',
                'Excellent leadership',
                'Good work-life balance',
                'Limited growth opportunities'
            ]
        })
        
        cls.employee_data = pd.DataFrame({
            'id': [1, 2, 3],
            'department_name': ['Technology', 'Sales', 'HR'],
            'position_title': ['Developer', 'Sales Rep', 'HR Analyst'],
            'gender': ['M', 'F', 'M'],
            'generation': ['Gen Y', 'Gen X', 'Gen Y'],
            'company_tenure': ['1-2 years', '2-5 years', '0-1 year']
        })
        
        cls.features = pd.DataFrame({
            'sentiment_score_feedback': [0.2, 0.6, -0.2],
            'confidence_feedback': [0.85, 0.88, 0.72],
            'text_feature_1': [0.5, 0.6, 0.3]
        }, index=[1, 2, 3])
        
        cls.clusters = np.array([0, 0, 1])
        
        cls.persona_generator = PersonaGenerator(
            n_terms=3,
            min_term_freq=1
        )
    
    def test_extract_key_terms(self):
        """Test key terms extraction from text"""
        text_data = self.sentiment_results['original_text'].tolist()
        key_terms = self.persona_generator.extract_key_terms(text_data)
        
        self.assertEqual(len(key_terms), 3)
        
        self.assertTrue(all(isinstance(term, str) for term in key_terms))
        
        self.assertTrue(any('environment' in term.lower() for term in key_terms))
        
        self.assertEqual(len(self.persona_generator.extract_key_terms([])), 0)
        self.assertEqual(len(self.persona_generator.extract_key_terms([None, ''])), 0)
    
    def test_calculate_sentiment_profile(self):
        """Test sentiment profile calculation"""
        cluster_comments = self.sentiment_results[
            self.sentiment_results['employee_id'].isin([1, 2])
        ]
        
        profile = self.persona_generator.calculate_sentiment_profile(cluster_comments)
        
        required_metrics = [
            'positive_ratio',
            'negative_ratio',
            'neutral_ratio',
            'avg_confidence',
            'avg_sentiment_score'
        ]
        for metric in required_metrics:
            self.assertIn(metric, profile)
        
        ratio_sum = (profile['positive_ratio'] + 
                    profile['negative_ratio'] + 
                    profile['neutral_ratio'])
        self.assertAlmostEqual(ratio_sum, 1.0, places=5)
        
        self.assertTrue(all(0 <= profile[key] <= 1 
                          for key in ['positive_ratio', 'negative_ratio', 'neutral_ratio', 'avg_confidence']))
    
    def test_analyze_demographics(self):
        """Test demographic analysis"""
        cluster_employees = self.employee_data[self.employee_data['id'].isin([1, 2])]
        demographics = self.persona_generator.analyze_demographics(cluster_employees)
        
        required_categories = [
            'departments',
            'positions',
            'generations',
            'genders',
            'tenures'
        ]
        for category in required_categories:
            self.assertIn(category, demographics)
        
        for category, distribution in demographics.items():
            self.assertIsInstance(distribution, dict)
            self.assertTrue(all(isinstance(count, int) for count in distribution.values()))
            self.assertTrue(sum(distribution.values()) == len(cluster_employees))
    
    def test_calculate_feature_averages(self):
        """Test feature average calculation"""
        cluster_features = self.features[self.clusters == 0]
        averages = self.persona_generator.calculate_feature_averages(cluster_features)
        
        self.assertTrue(any('sentiment_score' in key for key in averages.keys()))
        
        self.assertTrue(all(isinstance(val, float) for val in averages.values()))
        self.assertTrue(all(-1 <= val <= 1 for val in averages.values()))
    
    def test_generate_personas_integration(self):
        """Test complete persona generation pipeline"""
        personas = self.persona_generator.generate_personas(
            self.clusters,
            self.features,
            self.sentiment_results,
            self.employee_data
        )
        
        self.assertEqual(len(personas), len(np.unique(self.clusters)))
        
        for cluster_id, persona in personas.items():
            required_fields = [
                'size',
                'demographics',
                'sentiment_profile',
                'key_terms',
                'avg_scores'
            ]
            for field in required_fields:
                self.assertIn(field, persona)
            
            self.assertIsInstance(persona['size'], int)
            self.assertIsInstance(persona['demographics'], dict)
            self.assertIsInstance(persona['sentiment_profile'], dict)
            self.assertIsInstance(persona['key_terms'], list)
            self.assertIsInstance(persona['avg_scores'], dict)

if __name__ == '__main__':
    unittest.main(verbosity=2)