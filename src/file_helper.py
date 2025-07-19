# Utils/file_helper.py
# Utility functions for file operations, including JSON loading

import json
def load_json_data(file_path, encoding = "utf-8"):
    """Read JSON file and assign contents to a variable."""
    try:
        with open(file_path, 'r', encoding= encoding) as file:
            data = json.load(file)
        print(f"Successfully loaded JSON data from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return None