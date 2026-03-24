import pandas as pd
import sqlite3
from cryptography.fernet import Fernet

# --- 1. LOAD THE KEY AND SETUP FUNCTIONS ---
try:
    with open("secret.key", "rb") as key_file:
        key = key_file.read()
except FileNotFoundError:
    print("FATAL ERROR: secret.key not found. Cannot proceed.")
    exit()

fernet = Fernet(key)
db_file = 'secure_context.db'
table_name = 'secure_context_table'

def encrypt_value(value):
    return fernet.encrypt(str(value).encode('utf-8'))

def decrypt_value(encrypted_value):
    if encrypted_value is None: return None
    return fernet.decrypt(encrypted_value).decode('utf-8')

# --- 2. MODIFIED FUNCTIONS FOR CRUD OPERATIONS ---

def is_valid_company_id(company_id):
    """Checks if the company_id is a non-empty string."""
    if not isinstance(company_id, str) or not company_id.strip():
        # It's not a string OR it's an empty/whitespace-only string
        print(f"Error: company_ID must be a non-empty string, but received: {company_id} (type: {type(company_id).__name__})")
        return False
    return True


def create_record(data_dict):
    """Creates a new record. 'company_ID' must be in the data_dict."""
    if 'company_ID' not in data_dict:
        print("Error: 'company_ID' is required to create a new record.")
        return

    company_id_value = data_dict['company_ID']
    # --- VALIDATION ---
    if not is_valid_company_id(company_id_value):
        return
    # --------------------

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # company_id_value = data_dict['company_ID']
    columns_to_encrypt = [col for col in data_dict.keys() if col != 'company_ID']
    encrypted_values = [encrypt_value(data_dict[col]) for col in columns_to_encrypt]
    
    values_to_insert = [company_id_value] + encrypted_values
    all_columns_sql = ", ".join([f'"{c}"' for c in ['company_ID'] + columns_to_encrypt])
    placeholders = ", ".join(["?"] * len(values_to_insert))

    insert_sql = f"INSERT INTO {table_name} ({all_columns_sql}) VALUES ({placeholders})"
    
    try:
        cursor.execute(insert_sql, values_to_insert)
        conn.commit()
        print("New record created successfully.")
    except sqlite3.IntegrityError:
        print(f"Error: A record with company_ID '{company_id_value}' already exists.")
    finally:
        conn.close()


def get_all_data_as_dataframe():
    """Reads all data, decrypts it, and returns a pandas DataFrame."""
    conn = sqlite3.connect(db_file)
    # This makes fetching rows as dictionaries easier
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    decrypted_data = []
    for row in rows:
        record = {}
        # 'row.keys()' gives us all column names, including company_ID
        for key in row.keys():
            if key == 'company_ID':
                record[key] = row[key] # Keep plaintext company_ID
            else:
                record[key] = decrypt_value(row[key]) # Decrypt others
        decrypted_data.append(record)
        
    conn.close()
    return pd.DataFrame(decrypted_data)


def update_record(company_id, column_to_update, new_value):
    """Updates a specific cell in the database, finding the record by company_ID."""
    # --- VALIDATION ---
    if not is_valid_company_id(company_id):
        return
    # --------------------
    encrypted_value = encrypt_value(new_value)
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    update_sql = f'UPDATE {table_name} SET "{column_to_update}" = ? WHERE "company_ID" = ?'
    cursor.execute(update_sql, (encrypted_value, company_id))
    
    conn.commit()
    # Check if any row was actually updated
    if cursor.rowcount == 0:
        print(f"Warning: No record found with company_ID '{company_id}'. Nothing updated.")
    else:
        print(f"Record '{company_id}' updated successfully.")
    conn.close()


def delete_record(company_id):
    """Deletes a record by its company_ID."""
    # --- VALIDATION ---
    if not is_valid_company_id(company_id):
        return
    # --------------------
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute(f'DELETE FROM {table_name} WHERE "company_ID" = ?', (company_id,))
    
    conn.commit()
    if cursor.rowcount == 0:
        print(f"Warning: No record found with company_ID '{company_id}'. Nothing deleted.")
    else:
        print(f"Record '{company_id}' deleted successfully.")
    conn.close()


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    print("--- Reading all data for AI system ---")
    my_ai_dataframe = get_all_data_as_dataframe()
    print(my_ai_dataframe.head())

    # print("\n--- Updating a record ---")
    # # Example: Update the 'Salary' for the company with ID 'COMP-001'
    # update_record(company_id='COMP-001', column_to_update='Salary', new_value=150000)
    
    # print("\n--- Deleting a record ---")
    # delete_record(company_id='COMP-003')

    # print("\n--- Verifying changes ---")
    # final_df = get_all_data_as_dataframe()
    # print(final_df)