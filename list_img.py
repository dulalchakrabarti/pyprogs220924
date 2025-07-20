import os

def list_all_recursively(start_path):
    """
    Recursively lists all folders, subfolders, and files starting from a given path.

    Args:
        start_path (str): The path from which to start the listing.
    """
    if not os.path.isdir(start_path):
        print(f"Error: '{start_path}' is not a valid directory.")
        return

    print(f"Listing contents of: {start_path}\n")

    for dirpath, dirnames, filenames in os.walk(start_path):
        # Print current directory
        print(f"Directory: {dirpath}")

        # Print subdirectories
        if dirnames:
            print("  Subdirectories:")
            for dirname in dirnames:
                print(f"    - {os.path.join(dirpath, dirname)}")

        # Print files
        if filenames:
            print("  Files:")
            for filename in filenames:
                print(f"    - {os.path.join(dirpath, filename)}")
        print("-" * 30) # Separator for better readability

# Example usage:
if __name__ == "__main__":
    # Get the current directory of the script for demonstration
    current_directory = os.getcwd()

    # You can change this to any path you want to list
    # For example:
    target_path = "/home/dc/cntk/wheatcam16072025/"
    #target_path = current_directory

    list_all_recursively(target_path)

    #print("\n--- Another example with a non-existent path ---")
    #list_all_recursively("/non/existent/path/123")
