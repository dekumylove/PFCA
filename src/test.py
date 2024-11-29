import torch
import json


data = '{"m": "5000"}'
data = json.loads(data)
print(data.get('m', 0))