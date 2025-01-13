from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

class PersonaGenerator:
    """
    A class to generate detailed personas from employee clusters.
    
    Attributes:
        n_terms (int): Number of key terms to extract per cluster
        min_term_freq (int): Minimum frequency for terms to be considered
        vectorizer (CountVectorizer): Vectorizer for text analysis
    """
    
    def __init__(
        self,
        n_terms: int = 5,
        min_term_freq: int = 2,
        stop_words: Optional[List[str]] = None
    ):
        """
        Initialize the PersonaGenerator.
        
        Args:
            n_terms: Number of key terms to extract per cluster
            min_term_freq: Minimum frequency for terms to be considered
            stop_words: Additional stop words to filter out
        """
        self.n_terms = n_terms
        self.min_term_freq = min_term_freq
        self.vectorizer = CountVectorizer(
            stop_words=stop_words or 'english',
            ngram_range=(1, 2),
            min_df=min_term_freq
        )
    
    def extract_key_terms(self, text_data: List[str]) -> List[str]:
        """
        Extract key terms from text data.
        
        Args:
            text_data: List of text comments
            
        Returns:
            List of key terms ordered by importance
        """
        if not text_data or not any(text_data):
            return []
        
        text = ' '.join(str(t).lower() for t in text_data if pd.notna(t))
        if not text.strip():
            return []
        
        try:
            vectorizer = CountVectorizer(
                stop_words='english',
                ngram_range=(1, 2),  
                min_df=1,           
                max_df=1.0,         
                token_pattern=r'(?u)\b\w+\b'  
            )
            
            augmented_text = []
            
            augmented_text.append(text)
            
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            augmented_text.extend(sentences)
            
            final_text = ' . '.join(augmented_text)
            
            term_matrix = vectorizer.fit_transform([final_text])
            terms = vectorizer.get_feature_names_out()
            term_freq = term_matrix.toarray()[0]
            
            term_scores = []
            for term, freq in zip(terms, term_freq):
                length_boost = len(term.split())  
                freq_score = freq * (1 + 0.5 * length_boost)
                
                if ' ' in term and any(part == term for part in term.split()):
                    freq_score *= 1.2
                    
                term_scores.append((term, freq_score))
            
            term_scores.sort(key=lambda x: (-x[1], x[0]))
            
            selected_terms = []
            seen_words = set()
            
            for term, _ in term_scores:
                if len(selected_terms) >= self.n_terms:
                    break
                    
                term_words = set(term.split())
                if not term_words.intersection(seen_words):
                    selected_terms.append(term)
                    seen_words.update(term_words)
            
            return selected_terms
            
        except Exception as e:
            print(f"Error extracting terms: {str(e)}")
            return []
    
    def calculate_sentiment_profile(
        self,
        cluster_comments: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate sentiment statistics for a cluster.
        
        Args:
            cluster_comments: DataFrame containing sentiment analysis results
            
        Returns:
            Dictionary of sentiment statistics
        """
        return {
            'positive_ratio': (cluster_comments['sentiment_label'] == 'positive').mean(),
            'negative_ratio': (cluster_comments['sentiment_label'] == 'negative').mean(),
            'neutral_ratio': (cluster_comments['sentiment_label'] == 'neutral').mean(),
            'avg_confidence': cluster_comments['confidence'].mean(),
            'avg_sentiment_score': cluster_comments['sentiment_score'].mean()
        }
    
    def analyze_demographics(
        self,
        cluster_employees: pd.DataFrame
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze demographic distribution in a cluster.
        
        Args:
            cluster_employees: DataFrame containing employee information
            
        Returns:
            Dictionary of demographic distributions
        """
        demographic_fields = {
            'departments': 'department_name',
            'positions': 'position_title',
            'generations': 'generation',
            'genders': 'gender',
            'tenures': 'company_tenure'
        }
        
        demographics = {}
        for key, field in demographic_fields.items():
            if field in cluster_employees.columns:
                demographics[key] = cluster_employees[field].value_counts().to_dict()
        
        return demographics
    
    def calculate_feature_averages(
        self,
        cluster_features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate average feature values for a cluster.
        
        Args:
            cluster_features: DataFrame containing feature values
            
        Returns:
            Dictionary of average feature values
        """
        sentiment_cols = [col for col in cluster_features.columns 
                        if 'sentiment_score' in col]
        
        return {
            col: cluster_features[col].mean()
            for col in sentiment_cols
            if abs(cluster_features[col].mean()) > 0.1 
        }
    
    def generate_personas(
        self,
        clusters: np.ndarray,
        features: pd.DataFrame,
        sentiment_results: pd.DataFrame,
        employee_data: pd.DataFrame
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate detailed personas for each cluster.
        
        Args:
            clusters: Array of cluster assignments
            features: DataFrame containing feature values
            sentiment_results: DataFrame containing sentiment analysis results
            employee_data: DataFrame containing employee information
            
        Returns:
            Dictionary mapping cluster IDs to persona information
        """
        try:
            print("Generating cluster personas...")
            personas = {}
            
            for cluster_id in range(len(np.unique(clusters))):
                print(f"\nAnalyzing cluster {cluster_id}...")
                cluster_mask = clusters == cluster_id
                
                cluster_employees = employee_data[
                    employee_data['id'].isin(features.index[cluster_mask])
                ]
                
                cluster_comments = sentiment_results[
                    sentiment_results['employee_id'].isin(features.index[cluster_mask])
                ]
                
                cluster_features = features[cluster_mask]
                
                key_terms = self.extract_key_terms(
                    cluster_comments['original_text'].dropna().tolist()
                )
                
                personas[cluster_id] = {
                    'size': int(cluster_mask.sum()),
                    'demographics': self.analyze_demographics(cluster_employees),
                    'sentiment_profile': self.calculate_sentiment_profile(cluster_comments),
                    'key_terms': key_terms,
                    'avg_scores': self.calculate_feature_averages(cluster_features)
                }
                
                if 'risk_score' in features.columns:
                    personas[cluster_id]['risk_metrics'] = {
                        'avg_risk': features[cluster_mask]['risk_score'].mean(),
                        'high_risk_ratio': (features[cluster_mask]['risk_score'] > 0.7).mean()
                    }
            
            self._print_persona_summaries(personas)
            
            return personas
            
        except Exception as e:
            print(f"Error generating personas: {str(e)}")
            raise
    
    def _print_persona_summaries(self, personas: Dict[int, Dict[str, Any]]) -> None:
        """
        Print summaries of generated personas.
        
        Args:
            personas: Dictionary of persona information
        """
        print("\nCluster Personas Summary:")
        for cluster_id, persona in personas.items():
            print(f"\nCluster {cluster_id} (Size: {persona['size']} employees)")
            
            print("\nKey Terms:")
            print(", ".join(persona['key_terms']))
            
            print("\nSentiment Profile:")
            sentiment = persona['sentiment_profile']
            print(f"Positive: {sentiment['positive_ratio']:.2f}")
            print(f"Negative: {sentiment['negative_ratio']:.2f}")
            print(f"Confidence: {sentiment['avg_confidence']:.2f}")
            
            print("\nTop Department:")
            if 'departments' in persona['demographics']:
                top_dept = max(
                    persona['demographics']['departments'].items(),
                    key=lambda x: x[1]
                )
                print(f"{top_dept[0]}: {top_dept[1]} employees")
            
            if 'risk_metrics' in persona:
                print("\nRisk Profile:")
                print(f"Average Risk: {persona['risk_metrics']['avg_risk']:.2f}")
                print(f"High Risk Ratio: {persona['risk_metrics']['high_risk_ratio']:.2f}")
    
    def export_persona_report(
        self,
        personas: Dict[int, Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Export persona information to a markdown report.
        
        Args:
            personas: Dictionary of persona information
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Employee Cluster Personas Report\n\n")
            
            for cluster_id, persona in personas.items():
                f.write(f"## Cluster {cluster_id}\n\n")
                
                f.write(f"### Size: {persona['size']} employees\n\n")
                
                f.write("### Key Characteristics\n")
                f.write("- **Key Terms**: " + ", ".join(persona['key_terms']) + "\n")
                
                f.write("\n### Sentiment Profile\n")
                sentiment = persona['sentiment_profile']
                f.write(f"- Positive Ratio: {sentiment['positive_ratio']:.2f}\n")
                f.write(f"- Negative Ratio: {sentiment['negative_ratio']:.2f}\n")
                f.write(f"- Average Confidence: {sentiment['avg_confidence']:.2f}\n")
                
                f.write("\n### Demographics\n")
                for category, distribution in persona['demographics'].items():
                    f.write(f"\n#### {category.title()}\n")
                    for key, value in sorted(
                        distribution.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]:
                        f.write(f"- {key}: {value}\n")
                
                if 'risk_metrics' in persona:
                    f.write("\n### Risk Profile\n")
                    f.write(f"- Average Risk: {persona['risk_metrics']['avg_risk']:.2f}\n")
                    f.write(f"- High Risk Ratio: {persona['risk_metrics']['high_risk_ratio']:.2f}\n")
                
                f.write("\n---\n\n")