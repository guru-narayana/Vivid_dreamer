import os
import shutil

def delete_dir_contents(dir_path):
    """Deletes all files and folders within a given directory.

    Args:
        dir_path (str): The path to the directory to clear.
    """

    for filename in os.listdir(dir_path):
        if filename == "save" or filename == "configs":
            continue
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
           os.remove(file_path)  # Remove files
        elif os.path.isdir(file_path):
           shutil.rmtree(file_path)  # Recursively remove subdirectories
# Example usage:
dir_to_clear = "\
outputs/multigaussiandreamer-vsd/A_rustic_wrought-iron_candle_holder@20240606-133230"
delete_dir_contents(dir_to_clear)

print(f"Contents of '{dir_to_clear}' have been deleted.")