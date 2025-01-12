import argparse
from pathlib import Path
import sys
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.clustering.analyzer.feature_engineering import FeatureEngineer
from src.clustering.analyzer.cluster_analysis import ClusterAnalyzer
from src.clustering.analyzer.risk_analysis import RiskAnalyzer
from src.clustering.analyzer.personas import PersonaGenerator
from src.clustering.analyzer.visualization import ClusterVisualizer
from src.clustering.analyzer.utils import DataLoader, DataValidator, ResultsExporter, Logger

class ClusteringAnalysis:
    """Main class to orchestrate the clustering analysis pipeline."""
    
    def __init__(
        self,
        db_path: str,
        output_dir: str,
        log_file: Optional[str] = None
    ):
        """
        Initialize the clustering analysis pipeline.
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory to save results
            log_file: Path to log file (optional)
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.logger = Logger(log_file)
        
        self.feature_engineer = FeatureEngineer()
        self.cluster_analyzer = ClusterAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.persona_generator = PersonaGenerator()
        self.visualizer = ClusterVisualizer()
        self.exporter = ResultsExporter(output_dir)
        
    def run_analysis(self):
        """Run the complete clustering analysis pipeline."""
        try:
            self.logger.log_step("Starting analysis")
            
            self.logger.log_step("Loading data")
            sentiment_results, employee_data = DataLoader.load_from_db(
                self.db_path,
                """SELECT * FROM sentiment_analysis""",
                """
                SELECT e.*, d.name as department_name, p.title as position_title
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                JOIN positions p ON e.position_id = p.id
                """
            )
            
            self.logger.log_step("Validating data")
            if not (DataValidator.validate_sentiment_data(sentiment_results) and 
                   DataValidator.validate_employee_data(employee_data)):
                raise ValueError("Data validation failed")
            
            self.logger.log_step("Creating features")
            features = self.feature_engineer.create_features(
                sentiment_results,
                employee_data
            )
            
            self.logger.log_step("Performing clustering")
            kmeans_model, clusters, scaled_features, metrics = self.cluster_analyzer.perform_clustering(
                features,
                plot_metrics=True
            )
            
            self.logger.log_step("Calculating risk scores")
            risk_scores, cluster_risks = self.risk_analyzer.calculate_risk_scores(
                features,
                sentiment_results,
                clusters
            )
            
            self.logger.log_step("Generating personas")
            personas = self.persona_generator.generate_personas(
                clusters,
                features,
                sentiment_results,
                employee_data
            )
            
            self.logger.log_step("Creating visualizations")
            importance_by_cluster = self.cluster_analyzer.get_cluster_feature_importance(
                features, clusters
            )
            self.visualizer.create_dashboard(
                scaled_features,
                clusters,
                risk_scores,
                employee_data,
                importance_by_cluster
            )
            
            self.logger.log_step("Exporting results")
            self.exporter.export_results(
                clusters,
                risk_scores,
                personas,
                features
            )
            
            self.logger.log_step("Analysis completed successfully")
            
        except Exception as e:
            self.logger.log_error("Analysis failed", e)
            raise

def main():
    """Main entry point for the clustering analysis."""
    parser = argparse.ArgumentParser(
        description="Run employee feedback clustering analysis"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.log_file:
        args.log_file = output_dir / "analysis.log"
    
    analyzer = ClusteringAnalysis(
        db_path=args.db_path,
        output_dir=output_dir,
        log_file=args.log_file
    )
    
    analyzer.run_analysis()

if __name__ == "__main__":
    main()