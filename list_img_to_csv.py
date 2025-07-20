import os
import pandas as pd
from pathlib import Path

def get_directory_structure(root_dir):
    data = []
    root_dir = Path(root_dir)
    
    for root, dirs, files in os.walk(root_dir):
        current_path = Path(root)
        
        # Add directories first
        for dir_name in dirs:
            #print(dir_name)
            dir_path = current_path / dir_name
            relative_path = dir_path.relative_to(root_dir)
            data.append({
                'name': dir_name,
                'path': str(relative_path),
                'type': 'directory',
                'size': '',
                'extension': ''
            })
        
        # Add files
        for file_name in files:
            #print(file_name)
            file_path = current_path / file_name
            relative_path = file_path.relative_to(root_dir)
            file_stats = file_path.stat()
            
            # Get file extension
            ext = os.path.splitext(file_name)[1].lower()
            
            data.append({
                'name': file_name,
                'path': str(relative_path),
                'type': 'file',
                'size': file_stats.st_size,
                'extension': ext
            })
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Specify the directory you want to scan
    target_directory = input("Enter the directory path to scan: ") or '.'
    
    # Get the directory structure as a DataFrame
    df = get_directory_structure('/home/dc/cntk/wheatcam16072025/')
    print(df)
    # Save to CSV
    output_file = 'frmr_id_fld.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Directory structure saved to {output_file}")
    print(f"Total items found: {len(df)}")
    print(f"Breakdown:")
    print(f"- Directories: {len(df[df['type'] == 'directory'])}")
    print(f"- Files: {len(df[df['type'] == 'file'])}")

