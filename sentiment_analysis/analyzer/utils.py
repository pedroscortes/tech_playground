import pandas as pd
from typing import Dict, List
import sqlite3
from pathlib import Path
import json

def load_feedback_data(db_path: Path) -> pd.DataFrame:
    """Load feedback data from SQLite database."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        fr.*,
        e.name as employee_name,
        d.name as department_name,
        p.title as position_title
    FROM feedback_responses fr
    JOIN employees e ON fr.employee_id = e.id
    JOIN departments d ON e.department_id = d.id
    JOIN positions p ON e.position_id = p.id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def save_sentiment_results(df: pd.DataFrame, db_path: Path) -> None:
    """Save sentiment analysis results to database with proper list handling."""
    conn = sqlite3.connect(db_path)
    
    df_processed = df.copy()
    
    df_processed['positive_words'] = df_processed['positive_words'].apply(lambda x: ','.join(x) if x else '')
    df_processed['negative_words'] = df_processed['negative_words'].apply(lambda x: ','.join(x) if x else '')
    
    df_processed['word_scores'] = df_processed['word_scores'].apply(json.dumps)
    
    df_processed.to_sql('sentiment_analysis', conn, if_exists='replace', index=False)
    
    print("Creating summary views...")
    
    conn.execute("""
    CREATE VIEW IF NOT EXISTS sentiment_summary_by_field AS
    SELECT 
        field,
        COUNT(*) as total_comments,
        AVG(normalized_score) as avg_sentiment,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
        AVG(confidence) as avg_confidence
    FROM sentiment_analysis
    GROUP BY field
    """)
    
    conn.execute("""
    CREATE VIEW IF NOT EXISTS sentiment_summary_by_department AS
    SELECT 
        department,
        COUNT(*) as total_comments,
        AVG(normalized_score) as avg_sentiment,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
        AVG(confidence) as avg_confidence
    FROM sentiment_analysis
    GROUP BY department
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_field ON sentiment_analysis(field)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_department ON sentiment_analysis(department)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_analysis(sentiment_label)")
    
    conn.commit()
    conn.close()

def print_analysis_summary(df: pd.DataFrame) -> None:
    """Print comprehensive analysis summary."""
    print("\nAnalysis Summary:")
    print("-" * 50)
    
    print(f"Total comments analyzed: {len(df)}")
    
    sentiment_dist = df['sentiment_label'].value_counts()
    print("\nOverall Sentiment Distribution:")
    for label, count in sentiment_dist.items():
        print(f"{label}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nAverage confidence: {df['confidence'].mean():.2f}")
    
    print("\nDepartment Summary:")
    dept_summary = df.groupby(['department', 'sentiment_label']).size().unstack(fill_value=0)
    dept_total = dept_summary.sum(axis=1)
    
    for dept in dept_summary.index:
        print(f"\n{dept}:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = dept_summary.loc[dept, sentiment]
            pct = count/dept_total[dept]*100
            print(f"{sentiment}: {count} ({pct:.1f}%)")
    
    print("\nField Summary:")
    field_summary = df.groupby(['field', 'sentiment_label']).size().unstack(fill_value=0)
    field_total = field_summary.sum(axis=1)
    
    for field in field_summary.index:
        print(f"\n{field}:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = field_summary.loc[field, sentiment]
            pct = count/field_total[field]*100
            print(f"{sentiment}: {count} ({pct:.1f}%)")