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
        # self.prompt_pool, self.response_pool, self.score_pool, self.style_pool = [], [], [], []
        self.datapool_data = pd.DataFrame([], columns = ["prompt", "response", "score", "style", "cat_tokens"])

    def convert_cat_tokens(self, cat_token, target_style):
        return self.tree_tokens[STYLE_MAPPING[target_style]][cat_token]

    def add_style(self, prompts: List[str], responses: List[str], scores: List[float], styles: List[str], orig_styles: List[str]):
        # self.prompt_pool.extend(prompts)
        # self.response_pool.extend(responses)
        # self.score_pool.extend(scores)
        # self.style_pool.extend(styles)

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

        # assert len(self.prompt_pool) == len(self.response_pool) == len(self.score_pool) == len(self.style_pool) == len(orig_styles), f"not equal size in prompts, responses, scores, labels, and original styles {len(self.prompt_pool)}, {len(self.response_pool)}, {len(self.score_pool)}, {len(self.style_pool)}, and {len(orig_styles)}"

        #TODO examine if we really need this?
        # # filter out instances with same target and source styles
        # self.prompt_pool, self.response_pool, self.style_pool, self.score_pool, orig_styles = [list(x) for x in zip(*[(i, j, k, z, h) for i,j,k,z,h in 
        #         zip(self.prompt_pool, self.response_pool, self.style_pool, self.score_pool, orig_styles) if k != h])]

        # # sort by scores
        # sorted_data = sorted(zip(self.prompt_pool, self.response_pool, self.style_pool, self.score_pool),
        #                      key=lambda x: x[-1], reverse=True)
        # self.prompt_pool, self.response_pool, self.style_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]

        # cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens) for i in range(self.n_extra_tokens)]
        # cat_pos = [y for x in cat_pos for y in x]

        # cat_pos = cat_pos + [self.n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))
        # assert len(cat_pos) == len(self.style_pool), f"list of control code doesn't have same length as list of styles: {len(cat_pos)} vs {len(styles)}"
        # self.cat_tokens = [self.tree_tokens[STYLE_MAPPING[st]][cp] for st, cp in zip(self.style_pool, cat_pos)]

    def get_data(self):
        # return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.cat_tokens)
        return self.datapool_data.prompt.tolist(), self.datapool_data.response.tolist(), self.datapool_data.cat_tokens.tolist()

