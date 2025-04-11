import json

import pandas as pd

with open('results/overall_results.jsonl', 'r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))

table = {}
for line in data:
    extraction_model = line['extraction_model']
    match_model = line['match_model']
    accuracy = line['accuracy (IoU > 0.5)']

    if extraction_model not in table:
        table[extraction_model] = {}
    table[extraction_model][match_model] = accuracy

df = pd.DataFrame.from_dict(table, orient='index')

df = df.round(3)

df.to_csv('results/overall_results.csv')

print(df)

