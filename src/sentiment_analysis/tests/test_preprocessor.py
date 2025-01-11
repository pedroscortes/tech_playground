import unittest
from analyzer.preprocessor import TextPreprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        """Initialize preprocessor for tests."""
        self.preprocessor = TextPreprocessor()

    def test_normalize_text(self):
        """Test text normalization."""
        test_cases = [
            ("TEXTO EM MAIÚSCULO", "texto em maiusculo"),
            ("açúcar café", "acucar cafe"),
            ("", ""),
            (None, ""),
            ("Olá, MUNDO!", "ola mundo"),
            ("texto   com    espaços", "texto com espacos")
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.preprocessor.normalize_text(input_text)
                self.assertEqual(result, expected)

    def test_get_ngrams(self):
        """Test n-gram generation."""
        test_cases = [
            (
                "o dia está bonito hoje",
                {
                    'bigrams': ['o dia', 'dia esta', 'esta bonito', 'bonito hoje'],
                    'trigrams': ['o dia esta', 'dia esta bonito', 'esta bonito hoje']
                }
            ),
            (
                "texto curto",
                {
                    'bigrams': ['texto curto'],
                    'trigrams': []
                }
            ),
            (
                "",
                {
                    'bigrams': [],
                    'trigrams': []
                }
            )
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.preprocessor.get_ngrams(input_text)
                self.assertEqual(result['bigrams'], expected['bigrams'])
                self.assertEqual(result['trigrams'], expected['trigrams'])

    def test_preprocess(self):
        """Test full preprocessing pipeline."""
        test_cases = [
            (
                "O ambiente de trabalho é excelente",
                {
                    'tokens': ['ambi', 'trabalh', 'excel'],
                    'original_text': 'o ambiente de trabalho e excelente'
                }
            ),
            (
                "Não estou satisfeito com o trabalho",
                {
                    'tokens': ['NOT_satisfeit', 'trabalh'],
                    'original_text': 'nao estou satisfeito com o trabalho'
                }
            ),
            (
                "Sinto falta de feedback",
                {
                    'tokens': ['sinto_falta', 'feedback'],
                    'original_text': 'sinto falta de feedback'
                }
            ),
            (
                "Não é bom e não está satisfatório",
                {
                    'tokens': ['NOT_bom', 'NOT_satisfatori'],
                    'original_text': 'nao e bom e nao esta satisfatorio'
                }
            ),
            (
                "",
                {
                    'tokens': [],
                    'original_text': ''
                }
            )
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.preprocessor.preprocess(input_text)
                self.assertEqual(result['tokens'], expected['tokens'])
                self.assertEqual(result['original_text'], expected['original_text'])
                self.assertIn('bigrams', result)
                self.assertIn('trigrams', result)

    def test_stopword_removal(self):
        """Test stopword removal while keeping important words."""
        test_cases = [
            ("não quero isso", ['NOT_quer']),
            ("muito bom trabalho", ['muit', 'bom', 'trabalh']),
            ("o a os as de da do", []),
            ("mais menos pouco", ['mais', 'menos', 'pouc']),
            ("Muito BOM trabalho", ['muit', 'bom', 'trabalh']),
            ("muito, bom! trabalho.", ['muit', 'bom', 'trabalh'])
        ]
        
        for input_text, expected_tokens in test_cases:
            with self.subTest(input_text=input_text):
                result = self.preprocessor.preprocess(input_text)
                self.assertEqual(result['tokens'], expected_tokens)

if __name__ == '__main__':
    unittest.main()