import pandas as pd
from pathlib import Path
from connection import get_db_connection

def create_schema(conn):
    """Create the database schema"""
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS organizational_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            n0_empresa TEXT NOT NULL,
            n1_diretoria TEXT NOT NULL,
            n2_gerencia TEXT NOT NULL,
            n3_coordenacao TEXT NOT NULL,
            n4_area TEXT NOT NULL,
            UNIQUE (n0_empresa, n1_diretoria, n2_gerencia, n3_coordenacao, n4_area)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
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
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER REFERENCES employees(id),
            response_date TIMESTAMP NOT NULL,
            position_interest INTEGER,
            contribution INTEGER,
            learning_development INTEGER,
            feedback_score INTEGER,
            manager_interaction INTEGER,
            career_clarity INTEGER,
            permanence_expectation INTEGER,
            enps_score INTEGER,
            position_interest_comment TEXT,
            contribution_comment TEXT,
            learning_development_comment TEXT,
            feedback_comment TEXT,
            manager_interaction_comment TEXT,
            career_clarity_comment TEXT,
            permanence_expectation_comment TEXT,
            enps_comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_employee ON feedback_responses(employee_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_date ON feedback_responses(response_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_employee_department ON employees(department_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_employee_position ON employees(position_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_employee_org_unit ON employees(org_unit_id)')
    
    conn.commit()
    print("Schema created successfully!")

def safe_convert(value):
    """Convert string values to integers safely"""
    if pd.isna(value) or value == '-':
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None

def import_data():
    """Import data from CSV file to SQLite database"""
    try:
        project_root = Path(__file__).parent.parent.parent
        data_file = project_root / 'data' / 'raw' / 'data.csv'
        
        print("Reading CSV file...")
        df = pd.read_csv(
            data_file,
            encoding='utf-8',
            sep=';',
            quotechar='"'
        )
        print("CSV file read successfully!")
        print(f"Found {len(df)} rows and {len(df.columns)} columns")
        print("\nColumns found:", df.columns.tolist())
        
        conn = get_db_connection()
        
        print("\nCreating database schema...")
        create_schema(conn)
        
        cursor = conn.cursor()
        
        print("\nImporting departments...")
        departments = df['area'].unique()
        for dept in departments:
            if pd.notna(dept):
                cursor.execute(
                    "INSERT OR IGNORE INTO departments (name) VALUES (?)",
                    (dept,)
                )
        
        print("Importing positions...")
        positions = df['cargo'].unique()
        for pos in positions:
            if pd.notna(pos):
                cursor.execute(
                    "INSERT OR IGNORE INTO positions (title) VALUES (?)",
                    (pos,)
                )
        
        print("Importing locations...")
        locations = df['localidade'].unique()
        for loc in locations:
            if pd.notna(loc):
                cursor.execute(
                    "INSERT OR IGNORE INTO locations (name) VALUES (?)",
                    (loc,)
                )
        
        print("Importing organizational units...")
        org_units = df[['n0_empresa', 'n1_diretoria', 'n2_gerencia', 
                       'n3_coordenacao', 'n4_area']].drop_duplicates()
        
        for _, row in org_units.iterrows():
            if pd.notna(row).all():
                cursor.execute("""
                    INSERT OR IGNORE INTO organizational_units 
                    (n0_empresa, n1_diretoria, n2_gerencia, n3_coordenacao, n4_area)
                    VALUES (?, ?, ?, ?, ?)
                """, tuple(row))
        
        print("Importing employees and feedback...")
        for _, row in df.iterrows():
            cursor.execute("SELECT id FROM departments WHERE name = ?", (row['area'],))
            dept_id = cursor.fetchone()[0]
            
            cursor.execute("SELECT id FROM positions WHERE title = ?", (row['cargo'],))
            pos_id = cursor.fetchone()[0]
            
            cursor.execute("SELECT id FROM locations WHERE name = ?", (row['localidade'],))
            loc_id = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT id FROM organizational_units 
                WHERE n0_empresa = ? 
                AND n1_diretoria = ? 
                AND n2_gerencia = ? 
                AND n3_coordenacao = ? 
                AND n4_area = ?
            """, (row['n0_empresa'], row['n1_diretoria'], row['n2_gerencia'], 
                 row['n3_coordenacao'], row['n4_area']))
            org_id = cursor.fetchone()[0]
            
            cursor.execute("""
                INSERT OR REPLACE INTO employees 
                (name, email, corporate_email, department_id, position_id, role,
                location_id, company_tenure, gender, generation, org_unit_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['nome'], row['email'], row['email_corporativo'],
                dept_id, pos_id, row['funcao'], loc_id,
                row['tempo_de_empresa'], row['genero'],
                row['geracao'], org_id
            ))
            
            employee_id = cursor.lastrowid
            
            cursor.execute("""
                INSERT INTO feedback_responses 
                (employee_id, response_date, position_interest, contribution,
                learning_development, feedback_score, manager_interaction,
                career_clarity, permanence_expectation, enps_score,
                position_interest_comment, contribution_comment,
                learning_development_comment, feedback_comment,
                manager_interaction_comment, career_clarity_comment,
                permanence_expectation_comment, enps_comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                employee_id,
                pd.to_datetime(row['Data da Resposta']).strftime('%Y-%m-%d %H:%M:%S'),
                safe_convert(row['Interesse no Cargo']),
                safe_convert(row['Contribuição']),
                safe_convert(row['Aprendizado e Desenvolvimento']),
                safe_convert(row['Feedback']),
                safe_convert(row['Interação com Gestor']),
                safe_convert(row['Clareza sobre Possibilidades de Carreira']),
                safe_convert(row['Expectativa de Permanência']),
                safe_convert(row['eNPS']),
                row['Comentários - Interesse no Cargo'],
                row['Comentários - Contribuição'],
                row['Comentários - Aprendizado e Desenvolvimento'],
                row['Comentários - Feedback'],
                row['Comentários - Interação com Gestor'],
                row['Comentários - Clareza sobre Possibilidades de Carreira'],
                row['Comentários - Expectativa de Permanência'],
                row['[Aberta] eNPS']
            ))
        
        conn.commit()
        print("Data import completed successfully!")
        
    except Exception as e:
        print(f"Error importing data: {e}")
        if 'conn' in locals():
            conn.rollback()
        raise
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import_data()