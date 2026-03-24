import pandas as pd
import sqlite3
from cryptography.fernet import Fernet

# --- 1. KEY GENERATION / LOADING ---
try:
    with open("secret.key", "rb") as key_file:
        key = key_file.read()
    print("Key already exists. Loading it.")
except FileNotFoundError:
    print("Key not found. Generating a new one.")
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)

# --- 2. SETUP ENCRYPTION ---
fernet = Fernet(key)

def encrypt_value(value):
    """Encrypts a single value after converting it to a string and encoding."""
    return fernet.encrypt(str(value).encode('utf-8'))

# --- 3. READ DATA FROM EXCEL ---
excel_file = 'Context.xlsx'
db_file = 'secure_context.db'
table_name = 'secure_context_table'

try:
    df = pd.read_excel(excel_file)
    print(f"Successfully read data from {excel_file}")
    # --- **NEW**: Check if company_ID exists ---
    if 'company_ID' not in df.columns:
        print("FATAL ERROR: 'company_ID' column not found in the Excel file. Please check the column name.")
        exit()
except FileNotFoundError:
    print(f"ERROR: {excel_file} not found!")
    exit()

# --- 4. CONNECT TO DATABASE AND CREATE TABLE ---
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# --- **MODIFIED**: Define table structure with company_ID as the PRIMARY KEY ---
# Get all columns that need to be encrypted (i.e., every column EXCEPT company_ID)
columns_to_encrypt = [col for col in df.columns if col != 'company_ID']
encrypted_column_definitions = ", ".join([f'"{col}" BLOB' for col in columns_to_encrypt])

# Create the table with company_ID as a TEXT primary key and others as BLOBs
# IMPORTANT: Assumes company_ID is text-like. If it's purely a number, you can use INTEGER PRIMARY KEY.
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    "company_ID" TEXT PRIMARY KEY,
    {encrypted_column_definitions}
)
"""
cursor.execute(f"DROP TABLE IF EXISTS {table_name}") # Clean slate
cursor.execute(create_table_sql)
print(f"Database table '{table_name}' created with 'company_ID' as the primary key.")

# --- 5. ITERATE, ENCRYPT, AND INSERT DATA ---
print("Starting data migration with encryption...")
for index, row in df.iterrows():
    # Get the plaintext company_ID
    company_id_value = row['company_ID']
    
    # Encrypt all other values
    encrypted_other_values = [encrypt_value(row[col_name]) for col_name in columns_to_encrypt]
    
    # Combine the plaintext ID with the encrypted values
    values_to_insert = [company_id_value] + encrypted_other_values
    
    # Prepare the INSERT statement
    all_columns_sql = ", ".join([f'"{c}"' for c in ['company_ID'] + columns_to_encrypt])
    placeholders = ", ".join(["?"] * (len(columns_to_encrypt) + 1))
    insert_sql = f'INSERT INTO {table_name} ({all_columns_sql}) VALUES ({placeholders})'
    
    cursor.execute(insert_sql, values_to_insert)

# --- 6. COMMIT AND CLOSE ---
conn.commit()
conn.close()

print("\nMigration Complete!")
print(f"Successfully encrypted and moved {len(df)} rows to '{table_name}'.")