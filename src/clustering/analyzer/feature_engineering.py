from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """
    A class to handle feature engineering for employee feedback clustering.
    
    Attributes:
        text_max_features (int): Maximum number of text features to extract
        tfidf_vectorizer (TfidfVectorizer): Vectorizer for text feature extraction
        scaler (StandardScaler): Scaler for numerical features
    """
    
    def __init__(self, text_max_features: int = 100):
        """
        Initialize the FeatureEngineer.
        
        Args:
            text_max_features (int): Maximum number of text features to extract
        """
        self.text_max_features = text_max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=text_max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        
    def create_sentiment_features(self, sentiment_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from sentiment analysis results.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            
        Returns:
            DataFrame containing sentiment-based features
        """
        sentiment_features = pd.pivot_table(
            sentiment_results,
            index='employee_id',
            columns='field',
            values=['sentiment_score', 'confidence', 'normalized_score'],
            aggfunc='first'
        ).fillna(0)
        
        sentiment_features.columns = [
            f"{col[0]}_{col[1]}" for col in sentiment_features.columns
        ]
        
        return sentiment_features
    
    def create_text_features(self, sentiment_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from text data using TF-IDF.
        
        Args:
            sentiment_results: DataFrame containing text data
            
        Returns:
            DataFrame containing text-based features
        """
        text_by_employee = sentiment_results.groupby('employee_id')['original_text'].apply(' '.join)
        text_features = pd.DataFrame(
            self.tfidf_vectorizer.fit_transform(text_by_employee).toarray(),
            index=text_by_employee.index,
            columns=[f'text_feature_{i}' for i in range(self.text_max_features)]
        )
        
        return text_features
    
    def create_temporal_features(self, sentiment_results: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from sentiment results.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            
        Returns:
            DataFrame containing temporal features
        """
        temporal_features = pd.DataFrame(index=sentiment_results['employee_id'].unique())
        
        temporal_features['sentiment_volatility'] = sentiment_results.groupby('employee_id')['sentiment_score'].agg(lambda x: x.std())
        
        temporal_features['feedback_frequency'] = sentiment_results.groupby('employee_id').size()
        
        temporal_features['avg_confidence'] = sentiment_results.groupby('employee_id')['confidence'].mean()
        
        return temporal_features.fillna(0)
    
    def create_categorical_features(self, employee_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from categorical employee data.
        
        Args:
            employee_data: DataFrame containing employee information
            
        Returns:
            DataFrame containing categorical features
        """
        categorical_features = pd.DataFrame(index=employee_data['id'])
        
        feature_columns = {
            'department': 'department_name',
            'position': 'position_title',
            'gender': 'gender',
            'generation': 'generation',
            'tenure': 'company_tenure'
        }
        
        for prefix, column in feature_columns.items():
            if column in employee_data.columns:
                dummies = pd.get_dummies(
                    employee_data.set_index('id')[column],
                    prefix=prefix
                )
                categorical_features = categorical_features.join(dummies)
        
        return categorical_features
    
    def create_features(self, sentiment_results: pd.DataFrame, employee_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for clustering analysis.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            employee_data: DataFrame containing employee information
            
        Returns:
            DataFrame containing all engineered features
        """
        try:
            sentiment_features = self.create_sentiment_features(sentiment_results)
            text_features = self.create_text_features(sentiment_results)
            temporal_features = self.create_temporal_features(sentiment_results)
            categorical_features = self.create_categorical_features(employee_data)
            
            final_features = pd.concat([
                sentiment_features,
                text_features,
                temporal_features,
                categorical_features
            ], axis=1).fillna(0)
            
            final_features = final_features.astype(float)
            
            print(f"\nCreated feature matrix with shape: {final_features.shape}")
            print("\nFeature groups:")
            print(f"- Sentiment features: {len([c for c in final_features.columns if 'sentiment' in c])}")
            print(f"- Text features: {len([c for c in final_features.columns if 'text_feature' in c])}")
            print(f"- Temporal features: {len([c for c in final_features.columns if any(x in c for x in ['volatility', 'frequency', 'confidence'])])}")
            print(f"- Categorical features: {len([c for c in final_features.columns if any(x in c for x in ['department_', 'position_', 'gender_', 'generation_', 'tenure_'])])}")
            
            return final_features
            
        except Exception as e:
            print(f"Error in feature creation: {str(e)}")
            raise