CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS organizational_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    n0_empresa TEXT NOT NULL,
    n1_diretoria TEXT NOT NULL,
    n2_gerencia TEXT NOT NULL,
    n3_coordenacao TEXT NOT NULL,
    n4_area TEXT NOT NULL,
    UNIQUE (n0_empresa, n1_diretoria, n2_gerencia, n3_coordenacao, n4_area)
);

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
);

CREATE TABLE IF NOT EXISTS feedback_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER REFERENCES employees(id),
    response_date TIMESTAMP NOT NULL,
    
    position_interest INTEGER CHECK (position_interest BETWEEN 1 AND 10),
    contribution INTEGER CHECK (contribution BETWEEN 1 AND 10),
    learning_development INTEGER CHECK (learning_development BETWEEN 1 AND 10),
    feedback_score INTEGER CHECK (feedback_score BETWEEN 1 AND 10),
    manager_interaction INTEGER CHECK (manager_interaction BETWEEN 1 AND 10),
    career_clarity INTEGER CHECK (career_clarity BETWEEN 1 AND 10),
    permanence_expectation INTEGER CHECK (permanence_expectation BETWEEN 1 AND 10),
    enps_score INTEGER CHECK (enps_score BETWEEN 0 AND 10),
    
    position_interest_comment TEXT,
    contribution_comment TEXT,
    learning_development_comment TEXT,
    feedback_comment TEXT,
    manager_interaction_comment TEXT,
    career_clarity_comment TEXT,
    permanence_expectation_comment TEXT,
    enps_comment TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_employee ON feedback_responses(employee_id);
CREATE INDEX IF NOT EXISTS idx_feedback_date ON feedback_responses(response_date);
CREATE INDEX IF NOT EXISTS idx_employee_department ON employees(department_id);
CREATE INDEX IF NOT EXISTS idx_employee_position ON employees(position_id);
CREATE INDEX IF NOT EXISTS idx_employee_org_unit ON employees(org_unit_id);