from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import logging
from datetime import datetime
import json

class DataLoader:
    """
    Utility class for loading and preprocessing data.
    """
    
    @staticmethod
    def load_from_db(
        db_path: Union[str, Path],
        sentiment_query: str,
        employee_query: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from SQLite database.
        
        Args:
            db_path: Path to SQLite database
            sentiment_query: SQL query for sentiment data
            employee_query: SQL query for employee data
            
        Returns:
            Tuple of (sentiment_results, employee_data) DataFrames
        """
        try:
            conn = sqlite3.connect(db_path)
            sentiment_results = pd.read_sql_query(sentiment_query, conn)
            employee_data = pd.read_sql_query(employee_query, conn)
            conn.close()
            
            return sentiment_results, employee_data
            
        except Exception as e:
            logging.error(f"Error loading data from database: {str(e)}")
            raise

class DataValidator:
    """
    Utility class for data validation and verification.
    """
    
    @staticmethod
    def validate_sentiment_data(df: pd.DataFrame) -> bool:
        """
        Validate sentiment analysis results DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = {
            'employee_id', 'sentiment_label', 'confidence',
            'sentiment_score', 'normalized_score', 'field'
        }
        
        return all(col in df.columns for col in required_columns)
    
    @staticmethod
    def validate_employee_data(df: pd.DataFrame) -> bool:
        """
        Validate employee data DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = {
            'id', 'department_name', 'position_title',
            'gender', 'generation', 'company_tenure'
        }
        
        return all(col in df.columns for col in required_columns)

class ResultsExporter:
    """
    Utility class for exporting analysis results.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize ResultsExporter.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_results(
        self,
        clusters: np.ndarray,
        risk_scores: pd.DataFrame,
        personas: Dict[int, Dict[str, Any]],
        features: pd.DataFrame
    ) -> None:
        """
        Export all analysis results.
        
        Args:
            clusters: Cluster assignments
            risk_scores: Risk score DataFrame
            personas: Persona dictionary
            features: Feature DataFrame
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cluster_df = pd.DataFrame({
            'employee_id': features.index,
            'cluster': clusters,
            'risk_score': risk_scores['overall_risk']
        })
        cluster_df.to_csv(
            self.output_dir / f'cluster_assignments_{timestamp}.csv',
            index=False
        )
        
        risk_scores.to_csv(
            self.output_dir / f'risk_scores_{timestamp}.csv'
        )
        
        with open(self.output_dir / f'personas_{timestamp}.json', 'w') as f:
            json.dump(personas, f, indent=2)

class MetricsCalculator:
    """
    Utility class for calculating various metrics.
    """
    
    @staticmethod
    def calculate_cluster_stability(
        features: pd.DataFrame,
        n_iterations: int = 10
    ) -> pd.DataFrame:
        """
        Calculate cluster stability through multiple iterations.
        
        Args:
            features: Feature DataFrame
            n_iterations: Number of clustering iterations
            
        Returns:
            DataFrame with stability metrics
        """
        from sklearn.cluster import KMeans
        
        stability_scores = []
        for i in range(n_iterations):
            kmeans = KMeans(n_clusters=3, random_state=i)
            clusters = kmeans.fit_predict(features)
            stability_scores.append(pd.Series(clusters))
        
        stability_df = pd.DataFrame(stability_scores).T
        
        return stability_df.apply(lambda x: x.value_counts().iloc[0] / len(x), axis=1)
    
    @staticmethod
    def calculate_silhouette_samples(
        features: np.ndarray,
        clusters: np.ndarray
    ) -> pd.Series:
        """
        Calculate silhouette scores for individual samples.
        
        Args:
            features: Feature array
            clusters: Cluster assignments
            
        Returns:
            Series of silhouette scores
        """
        from sklearn.metrics import silhouette_samples
        return pd.Series(
            silhouette_samples(features, clusters),
            index=range(len(clusters))
        )

class Logger:
    """
    Utility class for logging analysis steps and results.
    """
    
    def __init__(self, log_file: Optional[Union[str, Path]] = None):
        """
        Initialize Logger.
        
        Args:
            log_file: Path to log file
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_step(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        """
        Log an analysis step.
        
        Args:
            step_name: Name of the step
            details: Additional details to log
        """
        self.logger.info(f"Step: {step_name}")
        if details:
            self.logger.info(f"Details: {json.dumps(details, indent=2)}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """
        Log an error.
        
        Args:
            error_msg: Error message
            exception: Exception object if available
        """
        self.logger.error(error_msg)
        if exception:
            self.logger.error(f"Exception: {str(exception)}")

def get_default_queries() -> Dict[str, str]:
    """
    Get default SQL queries for data loading.
    
    Returns:
        Dictionary containing default queries
    """
    return {
        'sentiment': """
            SELECT * FROM sentiment_analysis
        """,
        'employee': """
            SELECT e.*, d.name as department_name, p.title as position_title
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            JOIN positions p ON e.position_id = p.id
        """
    }

def generate_analysis_id() -> str:
    """
    Generate a unique ID for the analysis run.
    
    Returns:
        Unique analysis ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"analysis_{timestamp}"