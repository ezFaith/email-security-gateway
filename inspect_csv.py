import pandas as pd

try:
    file_path = 'emails.csv'
    df = pd.read_csv(file_path, encoding='latin-1')
    
    print("Columns:")
    print(df.columns.tolist())
    
    print("\nFirst 5 rows:")
    print(df.head().to_string())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")
