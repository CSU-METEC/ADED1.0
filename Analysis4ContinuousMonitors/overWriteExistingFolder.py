import shutil
import os

def overwrite_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the existing folder
        shutil.rmtree(folder_path)
    # Create the folder
    os.makedirs(folder_path)
    #return folder
