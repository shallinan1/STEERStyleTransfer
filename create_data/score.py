import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'

from reward import Reward
import torch

# Score the data
class Scorer(Reward):
    def __init__(self, 
                 style_model_path: str, 
                 batch_size: int, 
                 device: int):
        
        save_path = None
        super().__init__(save_path, 
                         style_model_path, 
                         batch_size, 
                         device)
        print("SubClass init with hardcoded required_var")
    
    def get_scores(self, prompts, texts, target_styles):
        with torch.no_grad():
            style_predictions = self.style_clf([r.lower() for r in texts])
            acceptability_predictions = self.cola_clf(texts)
            sim_scores = self.sim_scorer(prompts, texts)

        style_prediction_probs = []
        for tgt_st, style_preds in zip(target_styles, style_predictions):
            # gold_label = 'LABEL_' + str(rev_clf_label_map[tgt_st])
            gold_label = tgt_st
            for item in style_preds:
                if item['label'] == gold_label:
                    style_prediction_probs.append(item['score'])
                    break

        return {
            "target_style_prob": style_prediction_probs,
            "acceptability_prob": [self.acceptability(a) for a in acceptability_predictions],
            "similarity": sim_scores
        }
