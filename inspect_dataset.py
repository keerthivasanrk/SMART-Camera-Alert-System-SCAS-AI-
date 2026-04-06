import pandas as pd
import os

xlsx_path = 'dataset/images_info.xlsx'
try:
    df = pd.read_excel(xlsx_path)
    print("Excel Head:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print(f"\nTotal Rows: {len(df)}")
    
    # Check if there's a column for File Path and Labels
    # We'll look for strings like 'path', 'image', 'file', 'name', 'label'
    potential_paths = [c for c in df.columns if any(k in c.lower() for k in ['path', 'file', 'image'])]
    potential_names = [c for c in df.columns if any(k in c.lower() for k in ['name', 'label', 'person'])]
    
    print(f"\nPotential Path Columns: {potential_paths}")
    print(f"Potential Name Columns: {potential_names}")
    
    if potential_paths and potential_names:
        # Check first 5 files if they exist
        for i, row in df.head(5).iterrows():
            img_path = str(row[potential_paths[0]])
            name = str(row[potential_names[0]])
            exists = os.path.exists(img_path)
            print(f"[{'Found' if exists else 'Not Found'}] {name}: {img_path}")
    
except Exception as e:
    print(f"Error reading Excel: {e}")
