import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd
import sqlite3
import time

class SentimentAnalyzer:
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Optimize for CPU if needed
        if self.device == "cpu":
            torch.set_num_threads(4)
    
    def analyze_text(self, text, max_length=512):
        """Analyze sentiment of a single text."""
        # Preprocess text
        if pd.isna(text) or not isinstance(text, str):
            return None
            
        # Tokenize and truncate
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return predictions[0].cpu().numpy()
    
    def analyze_batch(self, texts, batch_size=8):
        """Analyze sentiment for a batch of texts."""
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.analyze_text(text) for text in batch_texts]
            results.extend(batch_results)
            
            # Add a small delay to prevent CPU overload
            if self.device == "cpu":
                time.sleep(0.1)
        
        return results

def test_analyzer():
    # Test texts in Portuguese
    test_texts = [
        "Estou muito feliz com o ambiente de trabalho!",
        "O gestor não me dá feedback suficiente.",
        "O trabalho é okay, mas poderia ser melhor."
    ]
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test single text analysis
    print("\nTesting single text analysis:")
    result = analyzer.analyze_text(test_texts[0])
    print(f"Text: {test_texts[0]}")
    print(f"Sentiment scores: {result}")
    
    # Test batch analysis
    print("\nTesting batch analysis:")
    results = analyzer.analyze_batch(test_texts)
    
    for text, result in zip(test_texts, results):
        if result is not None:
            print(f"\nText: {text}")
            print(f"Sentiment scores: {result}")

if __name__ == "__main__":
    test_analyzer()