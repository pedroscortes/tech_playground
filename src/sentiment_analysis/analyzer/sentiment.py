from analyzer.preprocessor import TextPreprocessor
from typing import Dict, List, Union
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
        self.sentiment_dict = {
            'sinto_falta': -1.5,
            
            'promocao': 1.5,
            'promovido': 1.5,
            'carreira': 0.5,
            'crescimento': 1.5,
            'desenvolvimento': 1.5,
            'capacitacao': 1,
            'treinamento': 1,
            'mentoria': 1,
            'aprendizado': 1,
            'oportunidade': 1,
            'profissional': 0.5,
            
            'flexibilidade': 1,
            'equilibrio': 1,
            'cooperacao': 1,
            'colaboracao': 1,
            'respeito': 1,
            'transparencia': 1,
            'confianca': 1,
            'autonomia': 1,
            'remoto': 0.5,
            'hibrido': 0.5,
            'ambiente': 0.5,
            'cultura': 0.5,
            
            'lideranca': 0.5,
            'gestor': 0.5,
            'feedback': 0.5,
            'reconhecimento': 1.5,
            'valorizacao': 1.5,
            'valorizam': 1.5,
            'valorizado': 1.5,
            'suporte': 1,
            'apoio': 1,
            
            'beneficio': 1,
            'salario': 0.5,
            'remuneracao': 0.5,
            'bonus': 1,
            'plr': 1,
            'premio': 1,
            
            'falta': -1.5,
            'ausencia': -1,
            'insuficiente': -1,
            'limitado': -1,
            'desmotivado': -1.5,
            'insatisfeito': -1.5,
            'frustracao': -2,
            'frustrado': -2,
            'desorganizado': -1,
            'confuso': -1,
            'problematico': -1.5,
            'dificil': -1,
            'complicado': -1,
            'ruim': -1,
            'pessimo': -2,
            'deficiente': -1.5,
            'precario': -1.5,
            'estresse': -1.5,
            'pressao': -1,
            'sobrecarga': -1.5,
            'burnout': -2,
            'mal': -1,
            'evitar': -0.5,
            'entendido': -0.5,
            
            'melhorar': -0.5,
            'melhoraria': -0.5,
            'melhorada': -0.5,
            'poderia': -0.3,
            'poderiam': -0.3,
            'deveria': -0.3,
            'precisaria': -0.5,
            'necessita': -0.5,
            'faltaria': -1,
            
            'excelente': 2,
            'otimo': 2,
            'bom': 1,
            'positivo': 1,
            'satisfeito': 1.5,
            'contente': 1,
            'feliz': 2,
            'motivado': 1.5,
            'produtivo': 1.5,
            'inovador': 1.5,
            'eficiente': 1.5,
            'agradavel': 1,
            'organizado': 1,
            'estruturado': 1
        }
        
        self.stemmed_dict = {}
        for word, score in self.sentiment_dict.items():
            if '_' in word:  
                self.stemmed_dict[word] = score
            else:
                stemmed = self.preprocessor.stemmer.stem(
                    self.preprocessor.normalize_text(word)
                )
                self.stemmed_dict[stemmed] = score
        
        self.modifying_phrases = {
            'mas poderia': -0.5,
            'mas poderiam': -0.5,
            'pode ser': -0.3,
            'precisa ser': -0.5,
            'deve ser': -0.3
        }
        
        self.complete_phrases = {
            'excelente ambiente': 2,
            'otimo ambiente': 2,
            'bom ambiente': 1.5,
            'equipe colaborativa': 1.5,
            'boa comunicacao': 1.5,
            'ma comunicacao': -1.5,
            'falta comunicacao': -1.5,
            'sinto falta': -1.5,
            'pode melhorar': -0.5,
            'pode ser melhorada': -0.5,
            'precisa melhorar': -1,
        }
    
    def check_modifying_phrases(self, text: str) -> float:
        """Check for phrases that modify sentiment."""
        mod_score = 0
        text_lower = text.lower()
        
        for phrase, score in self.modifying_phrases.items():
            if phrase in text_lower:
                mod_score += score
        
        return mod_score
    
    def check_complete_phrases(self, text: str) -> float:
        """Check for complete phrases with sentiment."""
        phrase_score = 0
        text_lower = text.lower()
        
        for phrase, score in self.complete_phrases.items():
            if phrase in text_lower:
                phrase_score += score
        
        return phrase_score
    
    def analyze_text(self, text: str) -> Dict[str, Union[float, str, List[str], Dict[str, float]]]:
        """Analyze sentiment with improved balance and confidence calculation."""
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
        
        phrase_score = self.check_complete_phrases(text)
        mod_score = self.check_modifying_phrases(text)
        
        processed = self.preprocessor.preprocess(text)
        tokens = processed['tokens']
        
        if not tokens and phrase_score == 0:
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
        
        word_scores = {}
        for token in tokens:
            score = self.stemmed_dict.get(token, 0)
            if score != 0:
                word_scores[token] = score
        
        word_sentiment = sum(word_scores.values())
        total_sentiment = word_sentiment + phrase_score + mod_score
        
        token_weight = 0.7
        phrase_weight = 0.3
        
        normalized_score = (
            (word_sentiment * token_weight) +
            ((phrase_score + mod_score) * phrase_weight)
        ) / (1 if len(tokens) == 0 else len(tokens))
        
        sentiment_words = sum(1 for score in word_scores.values() if score != 0)
        word_coverage = sentiment_words / len(tokens) if tokens else 0
        
        avg_strength = (abs(total_sentiment) / 
                    (sentiment_words + (1 if phrase_score != 0 else 0))
                    if sentiment_words > 0 or phrase_score != 0 else 0)
        strength_factor = min(avg_strength / 2, 1.0)
        
        positive_count = sum(1 for score in word_scores.values() if score > 0)
        negative_count = sum(1 for score in word_scores.values() if score < 0)
        if phrase_score > 0:
            positive_count += 1
        elif phrase_score < 0:
            negative_count += 1
        
        total_sentiment_items = positive_count + negative_count
        if total_sentiment_items > 0:
            consistency = abs(positive_count - negative_count) / total_sentiment_items
        else:
            consistency = 0
        
        confidence = (
            word_coverage * 0.4 +
            strength_factor * 0.3 +
            consistency * 0.3
        )
        
        if total_sentiment_items == 0:
            sentiment_label = 'neutral'
        elif phrase_score < -1.0:  
            sentiment_label = 'negative'
        elif normalized_score > 0.05:
            sentiment_label = 'positive'
        elif normalized_score < -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        if positive_count > 0 and negative_count > 0:
            confidence *= 0.8  
        
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

    def analyze_batch(self, texts: List[str], batch_size: int = 100) -> List[Dict]:
        """Analyze sentiment for a batch of texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.analyze_text(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results