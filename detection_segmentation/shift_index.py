import os
import json

folder = os.path.dirname(__file__)
filename = os.path.join(folder, 'dataset/semantic_classes.json')
output = os.path.join(folder, 'dataset/semantic_classes_.json')
with open(filename, 'r') as f:
    info = json.load(f)
new_info = {}
for key, value in info.items():
    new_info[int(key)-1] = value

with open(output, 'w') as f:
    json.dump(new_info, f, indent=4)