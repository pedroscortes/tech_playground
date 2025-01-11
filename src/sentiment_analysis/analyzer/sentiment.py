from typing import Dict, List, Union
from .preprocessor import TextPreprocessor
import unicodedata

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
        self.sentiment_dict = {
            'positivo': 1,
            'excelente': 2,
            'otimo': 2,
            'bom': 1,
            'satisfeito': 1.5,
            'feliz': 2,
            'contente': 1,
            'eficiente': 1.5,
            'produtivo': 1.5,
            'motivado': 1.5,
            'colaborativo': 1.5,
            'equipe': 0.5,
            'ambiente': 0.5,
            'oportunidade': 1,
            'crescimento': 1,
            'desenvolvimento': 1,
            
            'negativo': -1,
            'ruim': -1.5,
            'pessimo': -2,
            'insatisfeito': -1.5,
            'triste': -1,
            'frustrado': -2,
            'desmotivado': -1.5,
            'problema': -1,
            'dificil': -1,
            'falta': -1.5,
            'mal': -1,
            'pior': -1.5,
            
            'muito': 1.2,
            'bastante': 1.2,
            'pouco': 0.8
        }
        
        self.stemmed_dict = {}
        for word, score in self.sentiment_dict.items():
            stemmed = self.preprocessor.stemmer.stem(
                self.preprocessor.normalize_text(word)
            )
            self.stemmed_dict[stemmed] = score
        
        self.complete_phrases = {
            'excelente ambiente': 2,
            'otimo ambiente': 2,
            'bom ambiente': 1.5,
            'equipe colaborativa': 1.5,
            'pessimo ambiente': -2,
            'ambiente ruim': -1.5,
            'sinto falta': -1.5,
            'muito bom': 1.5,
            'muito ruim': -1.5
        }
        
        self.modifying_phrases = {
            'pode melhorar': -0.5,
            'precisa melhorar': -0.8,
            'mas poderia': -0.5,
            'mas pode': -0.5
        }
    
    def check_phrases(self, text: str) -> tuple:
        """Check for both complete and modifying phrases."""
        normalized = self.preprocessor.normalize_text(text)
        
        phrase_score = 0
        for phrase, score in self.complete_phrases.items():
            if phrase in normalized:
                phrase_score += score
        
        mod_score = 0
        for phrase, score in self.modifying_phrases.items():
            if phrase in normalized:
                mod_score += score
        
        return phrase_score, mod_score
    
    def analyze_text(self, text: str) -> Dict[str, Union[float, str, List[str], Dict[str, float]]]:
        """Analyze sentiment with improved handling."""
        if not text or not isinstance(text, str):
            return {
                'sentiment_score': 0,
                'normalized_score': 0,
                'confidence': 0,
                'sentiment_label': 'neutral',
                'positive_words': [],
                'negative_words': [],
                'word_scores': {},
                'phrase_score': 0,
                'mod_score': 0
            }
        
        phrase_score, mod_score = self.check_phrases(text)
        
        processed = self.preprocessor.preprocess(text)
        tokens = processed['tokens']
        
        word_scores = {}
        for token in tokens:
            if token.startswith('NOT_'):
                original_token = token[4:]
                if original_token in self.stemmed_dict:
                    word_scores[token] = -self.stemmed_dict[original_token] * 1.5
            else:
                word_scores[token] = self.stemmed_dict.get(token, 0)
        
        word_sentiment = sum(word_scores.values())
        total_sentiment = word_sentiment + phrase_score + mod_score
        
        positive_count = sum(1 for score in word_scores.values() if score > 0)
        negative_count = sum(1 for score in word_scores.values() if score < 0)
        if phrase_score > 0: positive_count += 1
        if phrase_score < 0: negative_count += 1
        if mod_score < 0: negative_count += 1
        
        token_count = max(len(tokens), 1)
        normalized_score = total_sentiment / token_count
        
        sentiment_words = sum(1 for score in word_scores.values() if score != 0)
        coverage = sentiment_words / token_count if token_count > 0 else 0
        
        strength = abs(total_sentiment) / token_count if token_count > 0 else 0
        strength_factor = min(strength / 2, 1.0)
        
        total_indicators = positive_count + negative_count
        if total_indicators > 0:
            consistency = abs(positive_count - negative_count) / total_indicators
        else:
            consistency = 0
        
        confidence = (coverage * 0.4 + strength_factor * 0.3 + consistency * 0.3)
        
        if abs(phrase_score) >= 1.5:
            confidence = max(confidence, 0.4)
        if abs(total_sentiment) >= 2:
            confidence = max(confidence, 0.4)
            
        if positive_count > 0 and negative_count > 0:
            confidence *= 0.8
        
        if confidence < 0.2:
            sentiment_label = 'neutral'
        elif normalized_score > 0.1:
            sentiment_label = 'positive'
        elif normalized_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        if phrase_score <= -1.5 or any(token.startswith('NOT_') for token in tokens):
            sentiment_label = 'negative'
        if phrase_score >= 1.5:
            sentiment_label = 'positive'
        
        return {
            'sentiment_score': total_sentiment,
            'normalized_score': normalized_score,
            'confidence': confidence,
            'sentiment_label': sentiment_label,
            'positive_words': [w for w, s in word_scores.items() if s > 0],
            'negative_words': [w for w, s in word_scores.items() if s < 0],
            'word_scores': word_scores,
            'phrase_score': phrase_score,
            'mod_score': mod_score
        }