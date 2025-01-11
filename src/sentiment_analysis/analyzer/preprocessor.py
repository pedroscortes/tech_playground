import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import unicodedata
import pandas as pd
import re

class TextPreprocessor:
    def __init__(self):
        self.stemmer = RSLPStemmer()
        
        all_stopwords = set(stopwords.words('portuguese'))
        self.stop_words = all_stopwords - {
            'não', 'nao', 'muito', 'mais', 'menos', 
            'pouco', 'mas', 'sem', 'bom', 'bem'
        }
        
        self.tokenizer = RegexpTokenizer(r'\w+')
    
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
    
    def get_ngrams(self, text: str) -> dict:
        """Generate n-grams from normalized text."""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        return {
            'bigrams': bigrams,
            'trigrams': trigrams
        }
    
    def preprocess(self, text: str) -> dict:
        """Preprocess text with improved token and phrase handling."""
        normalized_text = self.normalize_text(text)
        
        ngrams = self.get_ngrams(text)
        
        tokens = normalized_text.split()
        
        processed_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token in {'nao', 'não'}:
                if i + 1 < len(tokens):
                    next_token = tokens[i+1]
                    if next_token not in self.stop_words or next_token in {'bom', 'bem'}:
                        stemmed = self.stemmer.stem(next_token)
                        processed_tokens.append(f"NOT_{stemmed}")
                        i += 2
                        continue
            
            elif token == 'sinto' and i + 1 < len(tokens) and tokens[i+1] == 'falta':
                processed_tokens.append('sinto_falta')
                i += 2
                continue
            
            elif token not in self.stop_words:
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)
            
            i += 1
        
        return {
            'tokens': processed_tokens,
            'bigrams': ngrams['bigrams'],
            'trigrams': ngrams['trigrams'],
            'original_text': normalized_text
        }