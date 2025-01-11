import unittest
from analyzer.sentiment import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        """Initialize analyzer for tests."""
        self.analyzer = SentimentAnalyzer()
    
    def test_basic_functionality(self):
        """Test basic sentiment analysis functionality."""
        result = self.analyzer.analyze_text("texto neutro")
        self.assertIsInstance(result, dict)
        required_keys = {
            'sentiment_score', 'normalized_score', 'confidence',
            'sentiment_label', 'positive_words', 'negative_words',
            'word_scores', 'phrase_score', 'mod_score'
        }
        self.assertEqual(set(result.keys()), required_keys)

    def test_positive_sentiment(self):
        """Test detection of positive sentiments."""
        test_cases = [
            (
                "Excelente ambiente de trabalho",
                {
                    'sentiment_label': 'positive',
                    'min_confidence': 0.4,
                    'min_score': 0.1
                }
            ),
            (
                "Equipe muito boa e ambiente colaborativo",
                {
                    'sentiment_label': 'positive',
                    'min_confidence': 0.4,
                    'min_score': 0.1
                }
            ),
            (
                "Muito satisfeito com o trabalho",
                {
                    'sentiment_label': 'positive',
                    'min_confidence': 0.4,
                    'min_score': 0.1
                }
            )
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                self.assertEqual(result['sentiment_label'], expected['sentiment_label'])
                self.assertGreaterEqual(result['confidence'], expected['min_confidence'])
                self.assertGreaterEqual(result['normalized_score'], expected['min_score'])

    def test_negative_sentiment(self):
        """Test detection of negative sentiments."""
        test_cases = [
            (
                "Ambiente muito ruim de trabalho",
                {
                    'sentiment_label': 'negative',
                    'min_confidence': 0.4,
                    'max_score': -0.1
                }
            ),
            (
                "Não estou satisfeito com o trabalho",
                {
                    'sentiment_label': 'negative',
                    'min_confidence': 0.4,
                    'max_score': -0.1
                }
            ),
            (
                "Sinto falta de reconhecimento",
                {
                    'sentiment_label': 'negative',
                    'min_confidence': 0.4,
                    'max_score': -0.1
                }
            )
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                self.assertEqual(result['sentiment_label'], expected['sentiment_label'])
                self.assertGreaterEqual(result['confidence'], expected['min_confidence'])
                self.assertLessEqual(result['normalized_score'], expected['max_score'])

    def test_neutral_sentiment(self):
        """Test detection of neutral sentiments."""
        test_cases = [
            ("O trabalho continua como antes", 'neutral'),
            ("Trabalhando no projeto novo", 'neutral'),
            ("", 'neutral'),
            ("Algumas coisas boas, outras ruins", 'neutral')
        ]
        
        for text, expected_label in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                self.assertEqual(result['sentiment_label'], expected_label)

    def test_mixed_sentiment_handling(self):
        """Test handling of mixed sentiments."""
        test_cases = [
            (
                "Bom ambiente, mas poderia melhorar",
                {
                    'max_confidence': 0.7,
                    'has_positive': True,
                    'has_negative': True
                }
            ),
            (
                "Falta comunicação apesar do bom ambiente",
                {
                    'max_confidence': 0.7,
                    'has_positive': True,
                    'has_negative': True
                }
            )
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                self.assertLessEqual(result['confidence'], expected['max_confidence'])
                has_positive = any(score > 0 for score in result['word_scores'].values())
                has_negative = (any(score < 0 for score in result['word_scores'].values()) 
                              or result['mod_score'] < 0)
                self.assertEqual(has_positive, expected['has_positive'])
                self.assertEqual(has_negative, expected['has_negative'])

    def test_phrase_detection(self):
        """Test detection and scoring of common phrases."""
        test_cases = [
            (
                "Excelente ambiente de trabalho",
                {'min_phrase_score': 0}
            ),
            (
                "Sinto falta de feedback",
                {'max_phrase_score': 0}
            ),
            (
                "Bom mas pode melhorar",
                {'mod_score': True}
            )
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                if 'min_phrase_score' in expected:
                    self.assertGreater(result['phrase_score'], expected['min_phrase_score'])
                if 'max_phrase_score' in expected:
                    self.assertLess(result['phrase_score'], expected['max_phrase_score'])
                if 'mod_score' in expected:
                    self.assertLess(result['mod_score'], 0)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        test_cases = [
            ("", {'sentiment_label': 'neutral', 'confidence': 0}),
            (None, {'sentiment_label': 'neutral', 'confidence': 0}),
            (".", {'sentiment_label': 'neutral', 'confidence': 0}),
            ("!!!???", {'sentiment_label': 'neutral', 'confidence': 0}),
            ("   ", {'sentiment_label': 'neutral', 'confidence': 0})
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                self.assertEqual(result['sentiment_label'], expected['sentiment_label'])
                self.assertEqual(result['confidence'], expected['confidence'])

    def test_confidence_scores(self):
        """Test that confidence scores are reasonable."""
        test_cases = [
            (
                "Excelente trabalho, muito bem feito",
                {'min_confidence': 0.4, 'sentiment': 'positive'}
            ),
            (
                "Péssimo ambiente de trabalho",
                {'min_confidence': 0.4, 'sentiment': 'negative'}
            ),
            (
                "Trabalho normal",
                {'max_confidence': 0.4, 'sentiment': 'neutral'}
            )
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.analyzer.analyze_text(text)
                self.assertEqual(result['sentiment_label'], expected['sentiment'])
                if 'min_confidence' in expected:
                    self.assertGreaterEqual(result['confidence'], expected['min_confidence'])
                if 'max_confidence' in expected:
                    self.assertLessEqual(result['confidence'], expected['max_confidence'])

if __name__ == '__main__':
    unittest.main()