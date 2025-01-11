from typing import Dict, List, Union, Tuple
from .preprocessor import TextPreprocessor

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
        }
        
        self.stemmed_dict = {}
        for word, score in self.sentiment_dict.items():
            stemmed = self.preprocessor.stemmer.stem(
                self.preprocessor.normalize_text(word)
            )
            self.stemmed_dict[stemmed] = score

    def check_phrases(self, text: str) -> Tuple[float, float]:
        """Check for phrases that modify sentiment."""
        normalized = self.preprocessor.normalize_text(text)
        
        positive_phrases = {
            'excelente ambiente': 2,
            'ambiente otimo': 1.5,
            'ambiente excelente': 2,
            'muito bom': 1.5,
            'equipe colaborativa': 1.5
        }
        
        negative_phrases = {
            'sinto falta': -1.5,
            'pessimo ambiente': -2,
            'ambiente ruim': -1.5,
            'muito ruim': -1.5,
            'precisa melhorar': -1,
            'falta de': -1.5
        }
        
        modifying_phrases = {
            'mas poderia': -0.5,
            'mas pode': -0.5,
            'pode melhorar': -0.5
        }
        
        phrase_score = 0
        for phrase, score in positive_phrases.items():
            if phrase in normalized:
                phrase_score += score
        
        for phrase, score in negative_phrases.items():
            if phrase in normalized:
                phrase_score += score
        
        mod_score = 0
        for phrase, score in modifying_phrases.items():
            if phrase in normalized:
                mod_score += score
        
        return phrase_score, mod_score
    
    def analyze_text(self, text: str) -> Dict[str, Union[float, str, List[str], Dict[str, float]]]:
            """Analyze sentiment with improved balance handling."""
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
            original_text = processed['original_text']
            
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
            
            positive_scores = [s for s in word_scores.values() if s > 0]
            negative_scores = [s for s in word_scores.values() if s < 0]
            if phrase_score > 0: positive_scores.append(phrase_score)
            if phrase_score < 0: negative_scores.append(phrase_score)
            if mod_score < 0: negative_scores.append(mod_score)
            
            token_count = max(len(tokens), 1)
            coverage = len([s for s in word_scores.values() if s != 0]) / token_count
            strength = sum(abs(s) for s in word_scores.values()) / token_count
            
            if abs(phrase_score) >= 1.5:
                coverage += 0.2
                strength += 0.2
            
            total_weight = abs(word_sentiment) + abs(phrase_score) + abs(mod_score)
            if total_weight > 0:
                normalized_score = total_sentiment / total_weight
            else:
                normalized_score = 0
            
            is_balanced = False
            pos_sentiment = sum(positive_scores)
            neg_sentiment = abs(sum(negative_scores))
            
            if positive_scores and negative_scores:
                sentiment_ratio = min(pos_sentiment, neg_sentiment) / max(pos_sentiment, neg_sentiment)
                is_balanced = (
                    sentiment_ratio > 0.7 or  
                    (abs(normalized_score) < 0.2 and len(positive_scores) == len(negative_scores))  
                )
            
            if 'boas' in original_text.lower() and 'ruins' in original_text.lower():
                sentiment_label = 'neutral'
                confidence = max(0.4, min(coverage * 0.7, 0.8))
            elif not word_scores and not phrase_score:
                sentiment_label = 'neutral'
                confidence = 0
            elif is_balanced:
                sentiment_label = 'neutral'
                confidence = max(0.4, min(coverage * 0.7, 0.8))
            elif is_balanced or abs(normalized_score) < 0.2:
                sentiment_label = 'neutral'
                confidence = max(0.4, min(coverage * 0.7, 0.8))
            elif normalized_score > 0.1:
                sentiment_label = 'positive'
                confidence = max(0.4, coverage * 0.6 + strength * 0.4)
            elif normalized_score < -0.1:
                sentiment_label = 'negative'
                confidence = max(0.4, coverage * 0.6 + strength * 0.4)
            else:
                sentiment_label = 'neutral'
                confidence = min(coverage * 0.8, 0.7)
            
            if positive_scores and negative_scores and not is_balanced:
                confidence *= 0.9
            if sentiment_label == 'neutral':
                confidence = min(confidence, 0.7)
            
            return {
                'sentiment_score': total_sentiment,
                'normalized_score': normalized_score,
                'confidence': min(confidence, 0.95),
                'sentiment_label': sentiment_label,
                'positive_words': [w for w, s in word_scores.items() if s > 0],
                'negative_words': [w for w, s in word_scores.items() if s < 0],
                'word_scores': word_scores,
                'phrase_score': phrase_score,
                'mod_score': mod_score
            }