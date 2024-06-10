from typing import List
from copy import deepcopy
import itertools
import operator

from utils.constants import STYLE_MAPPING
import pandas as pd
from IPython import embed



class DataPool:
    def __init__(self, tree_tokens, n_extra_tokens):
        self.tree_tokens = tree_tokens
        self.n_extra_tokens = n_extra_tokens

        self.cat_tokens = None
        self.datapool_data = pd.DataFrame([], columns = ["prompt", "response", "score", "style", "cat_tokens"])

    def convert_cat_tokens(self, cat_token, target_style):
        return self.tree_tokens[STYLE_MAPPING[target_style]][cat_token]

    def add_style(self, prompts: List[str], responses: List[str], scores: List[float], styles: List[str], orig_styles: List[str]):
        assert len(prompts) == len(responses) == len(scores) == len(styles) == len(orig_styles)
        self.datapool_data = pd.concat([self.datapool_data, pd.DataFrame({
            "prompt": prompts, "response": responses, "score": scores, "style": styles
        })], ignore_index=True)

        cat_indeces, cat_vals = [], []
        for _, temp_df in self.datapool_data.groupby("style"):
            sorted_df = temp_df.sort_values("score")
            cat_indeces.extend(sorted_df.index.tolist())
            cat_pos = [[i] * (len(sorted_df) // self.n_extra_tokens) for i in range(self.n_extra_tokens)]
            cat_pos = [y for x in cat_pos for y in x]
            if len(sorted_df) > len(cat_pos):
                print(len(sorted_df), len(cat_pos))
                cat_pos += cat_pos[-1] * (len(sorted_df) - len(cat_pos))
            cat_vals.extend(cat_pos)
        
        self.datapool_data.loc[cat_indeces, 'cat_tokens'] = cat_vals
        self.datapool_data["cat_tokens"] = self.datapool_data.apply(lambda x: self.convert_cat_tokens(x.cat_tokens, x.style), axis=1)

    def get_data(self):
        return self.datapool_data.prompt.tolist(), self.datapool_data.response.tolist(), self.datapool_data.cat_tokens.tolist()

