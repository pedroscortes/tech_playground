# Tech Playground - Employee Feedback Analysis

## Overview
This project implements a comprehensive employee feedback analysis system, focusing on data storage, exploratory analysis, sentiment analysis in Portuguese, and advanced clustering techniques for employee segmentation. The implementation uses Python, SQLite, and various machine learning libraries to derive actionable insights from employee survey responses.

## Selected Tasks Summary
From the available tasks, the following were implemented:
- [x] **Task 1**: SQLite Database Implementation
- [x] **Task 5**: Exploratory Data Analysis
- [x] **Task 10**: Sentiment Analysis for Portuguese Text
- [x] **Task 12**: Employee Clustering and Risk Analysis

## Project Structure
```tree
tech_playground/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── data.csv           # Employee feedback data
│   └── processed/              # Processed data
│       └── employee_feedback.db # SQLite database
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis
│   └── task_12_exploration.ipynb # Clustering Analysis (POC)
├── sql/
│   └── schema.sql             # Database schema
├── src/
│   ├── clustering/            # Employee clustering implementation
│   │   ├── analyzer/
│   │   │   ├── cluster_analysis.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── personas.py
│   │   │   ├── risk_analysis.py
│   │   │   ├── utils.py
│   │   │   └── visualization.py
│   │   ├── tests/            # Unit tests
│   │   └── main.py
│   ├── database/             # Database operations
│   │   ├── connection.py
│   │   ├── import_data.py
│   │   └── verify_import.py
│   └── sentiment_analysis/   # Sentiment analysis
│       ├── analyzer/
│       │   ├── preprocessor.py
│       │   ├── sentiment.py
│       │   └── utils.py
│       ├── tests/
│       └── main.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.9+
- SQLite3
- Git

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tech_playground.git
cd tech_playground
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Task Implementations

### Task 1: Database Implementation

#### Original Task Description
Design and implement a database to structure the data from the CSV file. Requirements include choosing an appropriate database system, designing a schema, writing import scripts, and ensuring data integrity.

#### Implementation Details
- **Database Choice**: SQLite
  - Chosen for portability and ease of setup
  - Suitable for the dataset size
  - No separate server required
  - Native Python support

#### Schema Design
```sql
CREATE TABLE departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    corporate_email TEXT UNIQUE NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    position_id INTEGER REFERENCES positions(id),
    role TEXT NOT NULL,
    location_id INTEGER REFERENCES locations(id),
    company_tenure TEXT NOT NULL,
    gender TEXT,
    generation TEXT,
    org_unit_id INTEGER REFERENCES organizational_units(id)
);

-- Additional tables and indexes defined in schema.sql
```

#### Technical Decisions

1. **Data Normalization**:
   - Separate tables for departments, positions, locations
   - Organizational hierarchy structure
   - Foreign key constraints for data integrity

2. **Performance Optimization**:
```sql
CREATE INDEX idx_feedback_employee ON feedback_responses(employee_id);
CREATE INDEX idx_feedback_date ON feedback_responses(response_date);
CREATE INDEX idx_employee_department ON employees(department_id);
```

3. **Data Import Pipeline**:
   - Transaction-based imports
   - Error handling and logging
   - Data validation steps

#### Running Instructions

1. Initialize database:
```bash
python -m src.database.import_data
```

2. Verify import:
```bash
python -m src.database.verify_import
```

### Task 5: Exploratory Data Analysis

#### Original Task Description
Analyze the dataset to extract meaningful insights, compute summary statistics, identify trends, and visualize key findings.

#### Implementation Details
- Comprehensive statistical analysis using pandas and numpy
- Visualization with matplotlib and seaborn
- Hypothesis testing for key metrics
- Correlation analysis

#### Technical Approach

1. **Data Loading and Cleaning**:
```python
query = """
SELECT 
    e.*, d.name as department, p.title as position,
    f.*
FROM employees e
JOIN departments d ON e.department_id = d.id
JOIN positions p ON e.position_id = p.id
JOIN feedback_responses f ON e.id = f.employee_id
"""
```

2. **Analysis Components**:
   - Demographic distribution
   - Satisfaction metrics
   - Department comparisons
   - Correlation analysis
   - Statistical testing

#### Running Instructions
```bash
jupyter notebook notebooks/eda.ipynb
```

### Task 10: Sentiment Analysis

#### Original Task Description
Perform sentiment analysis on Portuguese comment fields, with preprocessing, sentiment analysis, and result summarization.

#### Implementation Details

1. **Text Preprocessing**:
   - Custom Portuguese text normalization
   - Context-aware negation handling
   - Domain-specific stopwords

2. **Sentiment Analysis Engine**:
```python
class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_dict = {
            'positivo': 1,
            'excelente': 2,
            'otimo': 2,
            'bom': 1,
            'satisfeito': 1.5,
            'negativo': -1,
            'ruim': -1.5,
            'pessimo': -2,
            'insatisfeito': -1.5
        }
```

3. **Performance Features**:
   - Confidence scoring
   - Multi-field analysis
   - Detailed reporting
   - Context-aware sentiment modification

#### Running Instructions
```bash
python -m src.sentiment_analysis.main
```

### Task 12: Employee Clustering

#### Original Task Description
Creative exploration of the dataset using clustering techniques to identify employee patterns and risk factors.

#### Implementation Details

1. **Feature Engineering**:
   - Sentiment features (9 components)
   - Text features using TF-IDF (100 dimensions)
   - Temporal patterns (11 features)
   - Categorical encodings (24 features)

2. **Clustering Analysis**:
   - Optimal clusters: 9 (determined by validation metrics)
   - PCA for dimensionality reduction
   - Risk scoring system
   - Interactive visualizations

#### Key Findings

1. **Cluster Characteristics**:
```text
Cluster 7 (High Performers):
- Size: 16 employees
- Positive sentiment: 82%
- High confidence: 68%
- Key terms: "comunicação entre", "e colaborativa"
- Top Department: Balanced across all
- Generation: Equal Y and Z distribution

Cluster 5 (Risk Group):
- Size: 33 employees
- Negative sentiment: 53%
- Higher risk scores (0.35-0.45)
- Key terms: "há falta", "oportunidades"
- Generation: 76% Generation Z
- Position: 39% Interns

Cluster 8 (Largest Group):
- Size: 133 employees
- Positive sentiment: 45%
- Balanced risk profile
- Key terms: "empresa", "oferece", "ambiente"
- Even department distribution
- Mixed tenure profile
```

2. **Risk Distribution**:
   - High risk cluster (5): 0.35-0.45 risk score
   - Low risk cluster (7): 0.10-0.15 risk score
   - Average cluster risk: 0.15-0.25

3. **Validation Metrics**:
   - Silhouette Score: 0.12 for 9 clusters
   - Calinski-Harabasz Score: 19.5
   - Davies-Bouldin Score: 2.5

#### Running Instructions
```bash
python -m src.clustering.main --db-path data/processed/employee_feedback.db --output-dir results
```

## Testing

Run tests for each module:
```bash
# Run all tests
python -m unittest discover src -v

# Run specific test suite
python -m unittest src/clustering/tests/test_feature_engineering.py
```

## Future Improvements
1. Real-time analysis capabilities
   - Streaming data processing
   - Live dashboard updates
   - Automated alerts for risk patterns

2. Advanced NLP features
   - Deep learning models for text analysis
   - Topic modeling
   - Enhanced sentiment analysis

3. API implementation
   - RESTful endpoints
   - Authentication system
   - Rate limiting and caching

4. Automated reporting system
   - Scheduled reports
   - Custom dashboards
   - Email notifications