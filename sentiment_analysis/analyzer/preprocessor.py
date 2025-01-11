import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import pandas as pd
import unicodedata
import re
from typing import Dict, List, Tuple, Optional

class TextPreprocessor:
    def __init__(self):
        self.stemmer = RSLPStemmer()
        
        all_stopwords = set(stopwords.words('portuguese'))
        self.stop_words = all_stopwords - {
            'não', 'nao', 'muito', 'mais', 'menos', 
            'pouco', 'mas', 'sem', 'bom', 'bem', 'está',
            'estou', 'é', 'quero', 'quer'
        }
        
        self.negatable_words = {
            'bom', 'bem', 'satisfeito', 'satisfatório',
            'eficiente', 'produtivo', 'positivo', 'quero',
            'quer', 'gosto', 'gosta'
        }
        
        self.skip_words = {'é', 'esta', 'está', 'estou', 'estão', 'e', 'o', 'a', 'os', 'as'}
        
        self.tokenizer = RegexpTokenizer(r'\w+')
    
    def find_next_meaningful(self, tokens: List[str], start: int) -> Tuple[Optional[str], int]:
            """Find next meaningful token that can be negated."""
            i = start
            while i < len(tokens):
                if tokens[i] not in self.skip_words:
                    token = tokens[i]
                    stemmed = self.stemmer.stem(token)
                    
                    is_negatable = (
                        token in self.negatable_words or
                        stemmed in {self.stemmer.stem(w) for w in self.negatable_words} or
                        'satisf' in stemmed  
                    )
                    
                    if is_negatable:
                        return token, i
                i += 1
            return None, -1
    
    def normalize_text(self, text: str) -> str:
        """Remove accents, punctuation and convert to lowercase."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        text = re.sub(r'[^\w\s]', '', text)
        
        text = ' '.join(text.split())
        
        return text
    
    def get_ngrams(self, text: str) -> Dict[str, List[str]]:
        """Generate n-grams from normalized text."""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        return {
            'bigrams': bigrams,
            'trigrams': trigrams
        }
    
    def preprocess(self, text: str) -> Dict[str, List[str]]:
        """Preprocess text with improved token and phrase handling."""
        normalized_text = self.normalize_text(text)
        
        ngrams = self.get_ngrams(text)
        
        tokens = normalized_text.split()
        
        processed_tokens = []
        i = 0
        while i < len(tokens):
            current_token = tokens[i]
            
            if current_token in {'nao', 'não'}:
                next_token, next_idx = self.find_next_meaningful(tokens, i + 1)
                if next_token:
                    stemmed = self.stemmer.stem(next_token)
                    processed_tokens.append(f"NOT_{stemmed}")
                    i = next_idx + 1  
                    
                    if (i + 1 < len(tokens) and 
                        tokens[i] in {'nao', 'não', 'e'} and 
                        i + 2 < len(tokens) and
                        tokens[i+1] in {'esta', 'está'}):
                        next_token2, next_idx2 = self.find_next_meaningful(tokens, i + 2)
                        if next_token2:
                            stemmed2 = self.stemmer.stem(next_token2)
                            processed_tokens.append(f"NOT_{stemmed2}")
                            i = next_idx2 + 1
                    continue
            
            elif current_token == 'sinto' and i + 1 < len(tokens) and tokens[i+1] == 'falta':
                processed_tokens.append('sinto_falta')
                i += 2
                continue
            
            elif current_token not in self.stop_words:
                stemmed = self.stemmer.stem(current_token)
                processed_tokens.append(stemmed)
            
            i += 1
        
        return {
            'tokens': processed_tokens,
            'bigrams': ngrams['bigrams'],
            'trigrams': ngrams['trigrams'],
            'original_text': normalized_text
        }