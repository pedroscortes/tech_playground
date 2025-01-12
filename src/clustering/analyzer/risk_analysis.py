from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class RiskAnalyzer:
    """
    A class to handle risk analysis of employee feedback data.
    
    Attributes:
        risk_weights (Dict[str, float]): Weights for different risk components
        tenure_weights (Dict[str, float]): Weights for different tenure categories
    """
    
    def __init__(
        self,
        risk_weights: Optional[Dict[str, float]] = None,
        tenure_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the RiskAnalyzer.
        
        Args:
            risk_weights: Custom weights for risk components
            tenure_weights: Custom weights for tenure categories
        """
        self.risk_weights = risk_weights or {
            'sentiment_risk': 0.3,
            'confidence_risk': 0.2,
            'volatility_risk': 0.2,
            'engagement_risk': 0.15,
            'tenure_risk': 0.15
        }
        
        self.tenure_weights = tenure_weights or {
            'menos_1_ano': 1.0,    
            'entre_1_2_anos': 0.8, 
            'entre_2_5_anos': 0.6, 
            'entre_5_10_anos': 0.4,
            'mais_10_anos': 0.2    
        }
    
    def calculate_sentiment_risk(
        self,
        sentiment_results: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk based on sentiment analysis results.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            
        Returns:
            Series of sentiment risk scores by employee
        """
        employee_sentiments = sentiment_results.groupby('employee_id')['sentiment_label'].value_counts(normalize=True)
        negative_ratios = employee_sentiments.unstack().fillna(0)['negative']
        return negative_ratios
    
    def calculate_confidence_risk(
        self,
        sentiment_results: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk based on confidence scores.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            
        Returns:
            Series of confidence risk scores by employee
        """
        confidence_scores = sentiment_results.groupby('employee_id')['confidence'].mean()
        return 1 - confidence_scores
    
    def calculate_volatility_risk(
        self,
        sentiment_results: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk based on sentiment volatility.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            
        Returns:
            Series of volatility risk scores by employee
        """
        volatility = sentiment_results.groupby('employee_id')['normalized_score'].std().fillna(0)
        return volatility / volatility.max() if volatility.max() > 0 else volatility
    
    def calculate_engagement_risk(
        self,
        sentiment_results: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk based on feedback engagement.
        
        Args:
            sentiment_results: DataFrame containing sentiment analysis results
            
        Returns:
            Series of engagement risk scores by employee
        """
        feedback_counts = sentiment_results.groupby('employee_id').size()
        avg_feedback = feedback_counts.mean()
        return 1 - (feedback_counts / avg_feedback).clip(0, 1)
    
    def calculate_tenure_risk(
        self,
        features: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate risk based on employee tenure.
        
        Args:
            features: DataFrame containing employee features
            
        Returns:
            Series of tenure risk scores by employee
        """
        tenure_columns = [col for col in features.columns if col.startswith('tenure_')]
        if not tenure_columns:
            return pd.Series(0, index=features.index)
        
        tenure_risk = pd.Series(0, index=features.index)
        for col in tenure_columns:
            category = col.replace('tenure_', '')
            if category in self.tenure_weights:
                tenure_risk += features[col] * self.tenure_weights[category]
        
        return tenure_risk
    
    def calculate_risk_scores(
        self,
        features: pd.DataFrame,
        sentiment_results: pd.DataFrame,
        clusters: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate comprehensive risk scores.
        
        Args:
            features: DataFrame containing employee features
            sentiment_results: DataFrame containing sentiment analysis results
            clusters: Array of cluster assignments
            
        Returns:
            Tuple of (risk scores DataFrame, cluster risk summary DataFrame)
        """
        try:
            print("Calculating risk scores...")
            
            risk_scores = pd.DataFrame(index=features.index)
            
            risk_scores['sentiment_risk'] = self.calculate_sentiment_risk(sentiment_results)
            risk_scores['confidence_risk'] = self.calculate_confidence_risk(sentiment_results)
            risk_scores['volatility_risk'] = self.calculate_volatility_risk(sentiment_results)
            risk_scores['engagement_risk'] = self.calculate_engagement_risk(sentiment_results)
            risk_scores['tenure_risk'] = self.calculate_tenure_risk(features)
            
            risk_scores['overall_risk'] = sum(
                risk_scores[component] * weight
                for component, weight in self.risk_weights.items()
            )
            
            risk_scores['cluster'] = clusters
            
            cluster_risks = risk_scores.groupby('cluster').agg({
                'overall_risk': ['mean', 'std', 'min', 'max'],
                'sentiment_risk': 'mean',
                'confidence_risk': 'mean',
                'volatility_risk': 'mean',
                'engagement_risk': 'mean',
                'tenure_risk': 'mean'
            })
            
            print("\nRisk Analysis Summary:")
            print(f"Overall Risk - Mean: {risk_scores['overall_risk'].mean():.3f}")
            print(f"Overall Risk - Std: {risk_scores['overall_risk'].std():.3f}")
            print("\nCluster Risk Summary:")
            print(cluster_risks['overall_risk'].round(3))
            
            return risk_scores, cluster_risks
            
        except Exception as e:
            print(f"Error in risk calculation: {str(e)}")
            raise
    
    def identify_high_risk_clusters(
        self,
        cluster_risks: pd.DataFrame,
        threshold_std: float = 1.0
    ) -> List[int]:
        """
        Identify high-risk clusters based on mean risk scores.
        
        Args:
            cluster_risks: DataFrame containing cluster risk metrics
            threshold_std: Number of standard deviations above mean to consider high risk
            
        Returns:
            List of high-risk cluster IDs
        """
        mean_risks = cluster_risks['overall_risk']['mean']
        threshold = mean_risks.mean() + (mean_risks.std() * threshold_std)
        return mean_risks[mean_risks > threshold].index.tolist()
    
    def plot_risk_distribution(
        self,
        risk_scores: pd.DataFrame,
        cluster_risks: pd.DataFrame
    ) -> None:
        """
        Create visualizations of risk distributions.
        
        Args:
            risk_scores: DataFrame containing individual risk scores
            cluster_risks: DataFrame containing cluster risk metrics
        """
        fig1 = go.Figure()
        risk_components = list(self.risk_weights.keys())
        
        for cluster_id in risk_scores['cluster'].unique():
            cluster_means = [
                cluster_risks.loc[cluster_id, (component, 'mean')]
                for component in risk_components
            ]
            
            fig1.add_trace(go.Scatterpolar(
                r=cluster_means,
                theta=risk_components,
                name=f'Cluster {cluster_id}',
                fill='toself'
            ))
        
        fig1.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            title="Risk Components by Cluster",
            showlegend=True
        )
        fig1.show()
        
        fig2 = go.Figure()
        for cluster_id in risk_scores['cluster'].unique():
            cluster_risks = risk_scores[risk_scores['cluster'] == cluster_id]['overall_risk']
            
            fig2.add_trace(go.Box(
                y=cluster_risks,
                name=f'Cluster {cluster_id}',
                boxpoints='outliers'
            ))
        
        fig2.update_layout(
            title="Overall Risk Distribution by Cluster",
            yaxis_title="Risk Score",
            showlegend=True
        )
        fig2.show()