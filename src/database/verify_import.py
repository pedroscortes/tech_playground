from connection import get_db_connection

def verify_import():
    """Verify the imported data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        tables = ['departments', 'positions', 'locations', 'organizational_units', 
                 'employees', 'feedback_responses']
        
        print("Record counts:")
        print("-" * 40)
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table}: {count} records")
        
        print("\nDepartment distribution:")
        print("-" * 40)
        cursor.execute("""
            SELECT d.name, COUNT(e.id) as employee_count
            FROM departments d
            LEFT JOIN employees e ON d.id = e.department_id
            GROUP BY d.name
            ORDER BY employee_count DESC
        """)
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]} employees")
        
        print("\nAverage eNPS score by department:")
        print("-" * 40)
        cursor.execute("""
            SELECT d.name, 
                   ROUND(AVG(CAST(f.enps_score AS FLOAT)), 2) as avg_enps,
                   COUNT(f.id) as responses
            FROM departments d
            JOIN employees e ON d.id = e.department_id
            JOIN feedback_responses f ON e.id = f.employee_id
            GROUP BY d.name
            ORDER BY avg_enps DESC
        """)
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]} (from {row[2]} responses)")
            
    except Exception as e:
        print(f"Error verifying data: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    verify_import()