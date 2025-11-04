import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data")

def load_file(file: str):
    filepath = os.path.join(DATA_PATH, file)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data