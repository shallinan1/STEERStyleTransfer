"""
python3 -m create_data.create_from_nonparallel
"""
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'

import json
import pandas as pd
from create_data.score import Scorer

data_path = "create_data/test_nonparallel.json"
output_path = "create_data/from_nonparallel_example.jsonl"
filter_top_k = 2

with open(data_path, "r") as f:
    data = json.load(f)

# Create a DataFrame
df = pd.DataFrame(data)
df.rename(columns={'input': 'prompt', 'output': 'text'}, inplace=True)

scorer = Scorer(
    style_model_path = "hallisky/cds_style_classifier",
    batch_size=16,
    device=0
)

target_styles = ["LABEL_0"] # Placeholder for your label class - replace with your own class
scores = scorer.get_scores(df["prompt"].tolist(), df["text"].tolist(), target_styles * len(df))

df = pd.concat([df, pd.DataFrame(scores)], axis=1)
df["score"] = df["target_style_prob"] * df["similarity"] * df["acceptability_prob"]
df.sort_values(by="score", inplace=True,ascending=False)

save_df = df.groupby('target_style').head(filter_top_k)
save_df = save_df.sample(frac=1)

save_df.to_json(output_path, lines=True, orient='records')

