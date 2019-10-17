import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

with open(sys.argv[2], 'w') as f:
    json.dump({x['_id']: [[x['label'], 1]] for x in data}, f)
