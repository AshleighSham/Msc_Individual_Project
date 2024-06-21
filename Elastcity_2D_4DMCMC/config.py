import json
import os

def load_config(file_path):
    """
       Load a JSON configuration file.

       Args:
           file_path (str): The path to the JSON configuration file.

       Returns:
           dict: The loaded configuration as a dictionary.
       """
    with open(file_path, 'r') as file:
        return json.load(file)
    
path = os.path.join(os.path.dirname(__file__), 'config.json')
config = load_config(path)