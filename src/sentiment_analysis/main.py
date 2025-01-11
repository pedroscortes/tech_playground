from pathlib import Path
from analyzer.sentiment import SentimentAnalyzer
from analyzer.utils import load_feedback_data, save_sentiment_results, print_analysis_summary
import pandas as pd
from tqdm import tqdm

def main():
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / 'data' / 'processed' / 'employee_feedback.db'
    
    print(f"Using database at: {db_path}")
    
    print("\nLoading feedback data...")
    df = load_feedback_data(db_path)
    
    analyzer = SentimentAnalyzer()
    
    comment_fields = [
        'position_interest_comment',
        'contribution_comment',
        'learning_development_comment',
        'feedback_comment',
        'manager_interaction_comment',
        'career_clarity_comment',
        'permanence_expectation_comment',
        'enps_comment'
    ]
    
    all_results = []
    
    for field in comment_fields:
        print(f"\nAnalyzing {field}...")
        comments = df[df[field].str.strip() != '-'][field].dropna()
        print(f"Found {len(comments)} non-empty comments")
        
        field_results = []
        for idx, comment in tqdm(enumerate(comments), total=len(comments)):
            result = analyzer.analyze_text(comment)
            result.update({
                'field': field,
                'employee_id': df.iloc[comments.index[idx]]['employee_id'],
                'department': df.iloc[comments.index[idx]]['department_name'],
                'position': df.iloc[comments.index[idx]]['position_title'],
                'original_text': comment
            })
            field_results.append(result)
        
        positive = sum(1 for r in field_results if r['sentiment_label'] == 'positive')
        negative = sum(1 for r in field_results if r['sentiment_label'] == 'negative')
        neutral = sum(1 for r in field_results if r['sentiment_label'] == 'neutral')
        
        print(f"\nField Summary:")
        print(f"Positive: {positive} ({positive/len(field_results)*100:.1f}%)")
        print(f"Negative: {negative} ({negative/len(field_results)*100:.1f}%)")
        print(f"Neutral: {neutral} ({neutral/len(field_results)*100:.1f}%)")
        
        all_results.extend(field_results)
    
    print("\nSaving results to database...")
    results_df = pd.DataFrame(all_results)
    save_sentiment_results(results_df, db_path)
    
    print("\nOverall Analysis Summary:")
    print("-" * 50)
    print(f"Total comments analyzed: {len(all_results)}")
    
    print("\nDepartment Summary:")
    dept_summary = results_df.groupby(['department', 'sentiment_label']).size().unstack(fill_value=0)
    dept_total = dept_summary.sum(axis=1)
    for dept in dept_summary.index:
        print(f"\n{dept}:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = dept_summary.loc[dept, sentiment]
            pct = count/dept_total[dept]*100
            print(f"{sentiment}: {count} ({pct:.1f}%)")
    
    print("\nField Summary:")
    field_summary = results_df.groupby(['field', 'sentiment_label']).size().unstack(fill_value=0)
    field_total = field_summary.sum(axis=1)
    for field in field_summary.index:
        print(f"\n{field}:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = field_summary.loc[field, sentiment]
            pct = count/field_total[field]*100
            print(f"{sentiment}: {count} ({pct:.1f}%)")
    
    print("\nAnalysis complete! Results saved to database.")

if __name__ == "__main__":
    main()