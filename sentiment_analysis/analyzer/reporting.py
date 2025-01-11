import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

class SentimentReport:
    def __init__(self, db_path: Path):
        """Initialize reporter with database path."""
        self.db_path = db_path
    
    def get_summary_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get summary data from database."""
        conn = sqlite3.connect(self.db_path)
        
        results_df = pd.read_sql_query(
            "SELECT * FROM sentiment_analysis",
            conn
        )
        
        dept_summary = pd.read_sql_query(
            """
            SELECT 
                department,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                AVG(normalized_score) as avg_sentiment,
                AVG(confidence) as avg_confidence
            FROM sentiment_analysis
            GROUP BY department
            """,
            conn
        ).set_index('department')
        
        field_summary = pd.read_sql_query(
            """
            SELECT 
                field,
                COUNT(*) as total_comments,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                AVG(normalized_score) as avg_sentiment,
                AVG(confidence) as avg_confidence
            FROM sentiment_analysis
            GROUP BY field
            """,
            conn
        ).set_index('field')
        
        conn.close()
        return results_df, dept_summary, field_summary
    
    def print_enhanced_summary(self):
        """Print enhanced analysis summary with additional insights."""
        results_df, dept_summary, field_summary = self.get_summary_data()
        
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        print("\n1. OVERALL METRICS")
        print("-"*40)
        print(f"Total Comments Analyzed: {len(results_df)}")
        print(f"Average Confidence Score: {results_df['confidence'].mean():.2f}")
        
        sentiment_dist = results_df['sentiment_label'].value_counts()
        print("\nOverall Sentiment Distribution:")
        for label, count in sentiment_dist.items():
            print(f"{label.capitalize()}: {count} ({count/len(results_df)*100:.1f}%)")
        
        print("\n2. DEPARTMENT INSIGHTS")
        print("-"*40)
        for dept in dept_summary.index:
            stats = dept_summary.loc[dept]
            print(f"\n{dept.upper()}:")
            print(f"  • Sentiment Balance: {int(stats['positive_count'])}/{int(stats['neutral_count'])}/{int(stats['negative_count'])}")
            print(f"  • Average Sentiment: {stats['avg_sentiment']:.2f}")
            print(f"  • Average Confidence: {stats['avg_confidence']:.2f}")
        
        print("\n3. FEEDBACK CATEGORY ANALYSIS")
        print("-"*40)
        for field in field_summary.index:
            stats = field_summary.loc[field]
            total = stats['total_comments']
            pos_pct = stats['positive_count'] / total * 100
            neg_pct = stats['negative_count'] / total * 100
            neu_pct = stats['neutral_count'] / total * 100
            
            print(f"\n{field}:")
            print(f"  • Response Rate: {int(total)} comments")
            print(f"  • Sentiment Distribution:")
            print(f"    - Positive: {int(stats['positive_count'])} ({pos_pct:.1f}%)")
            print(f"    - Neutral:  {int(stats['neutral_count'])} ({neu_pct:.1f}%)")
            print(f"    - Negative: {int(stats['negative_count'])} ({neg_pct:.1f}%)")
            print(f"  • Average Sentiment: {stats['avg_sentiment']:.2f}")
            print(f"  • Average Confidence: {stats['avg_confidence']:.2f}")
        
        print("\n4. KEY INSIGHTS")
        print("-"*40)
        
        field_sentiments = field_summary['avg_sentiment'].sort_values()
        
        print("\nSentiment by Category:")
        print(f"• Most Positive: {field_sentiments.index[-1]}")
        print(f"  - Sentiment Score: {field_sentiments.iloc[-1]:.2f}")
        print(f"• Most Negative: {field_sentiments.index[0]}")
        print(f"  - Sentiment Score: {field_sentiments.iloc[0]:.2f}")
        
        dept_sentiments = dept_summary['avg_sentiment'].sort_values()
        
        print("\nDepartment Highlights:")
        print(f"• Highest Satisfaction: {dept_sentiments.index[-1].upper()}")
        print(f"  - Sentiment Score: {dept_sentiments.iloc[-1]:.2f}")
        print(f"• Lowest Satisfaction: {dept_sentiments.index[0].upper()}")
        print(f"  - Sentiment Score: {dept_sentiments.iloc[0]:.2f}")
        
        attention_threshold = 0.1 
        attention_fields = field_summary[
            field_summary['negative_count'] / field_summary['total_comments'] > attention_threshold
        ]
        
        if not attention_fields.empty:
            print("\nAreas Needing Attention:")
            for field in attention_fields.index:
                stats = attention_fields.loc[field]
                neg_pct = stats['negative_count'] / stats['total_comments'] * 100
                print(f"• {field}")
                print(f"  - {int(stats['negative_count'])} negative comments ({neg_pct:.1f}%)")
        
        high_conf_threshold = 0.8
        high_conf = results_df[results_df['confidence'] > high_conf_threshold]
        high_conf_dist = high_conf['sentiment_label'].value_counts()
        
        if len(high_conf) > 0:
            print("\nHigh Confidence Insights:")
            for label, count in high_conf_dist.items():
                pct = count / len(high_conf) * 100
                print(f"• {label.capitalize()}: {count} comments ({pct:.1f}% of high-confidence results)")
        
        print("\n" + "="*80)