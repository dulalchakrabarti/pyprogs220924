import os
from PIL import Image
import shutil

# Define source and destination root directories
src_root = 'path/to/source'
dest_root = 'path/to/destination'

# Traverse source directory and process JPGs
for root, _, files in os.walk(src_root):
    for file in files:
        if file.lower().endswith('.jpg'):
            # Full path to source image
            src_path = os.path.join(root, file)

            # Construct relative path and target subfolder
            rel_path = os.path.relpath(root, src_root)
            dest_folder = os.path.join(dest_root, rel_path)

            # Create destination subfolder if it doesn't exist
            os.makedirs(dest_folder, exist_ok=True)

            # Load image using PIL and save to destination
            img = Image.open(src_path)
            dest_path = os.path.join(dest_folder, file)
            img.save(dest_path)