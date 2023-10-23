import pandas as pd

# Load the CSV file
file_path = 'dev/tables_description.csv'
df = pd.read_csv(file_path)

# Check if 'embedding' column exists
if 'embedding' in df.columns:
    # Remove the 'embedding' column
    df = df.drop('embedding', axis=1)

    # Save the DataFrame back to CSV
    df.to_csv(file_path, index=False)
    print("Column 'embedding' has been removed and the file is saved.")
else:
    print("Column 'embedding' does not exist in the file.")

