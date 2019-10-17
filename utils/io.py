"""
Generic I/O utilities
"""

import json

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def write_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
