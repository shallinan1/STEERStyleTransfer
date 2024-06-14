import json
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
import math
import pandas as pd
from tqdm import tqdm
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Iterable, Dict, Any
import torch
from utils.similarity import compute_sim
from utils.utils import convert_label_list
from utils.utils import batchify, batchify_zip, load_jsonl
import json

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

# TODO Un-hardcode these?
with open("models/multilabel/styledict.json") as f:
    rev_clf_label_map = json.load(f)
clf_label_map = {v: k for k, v in rev_clf_label_map.items()}

class Reward:
    def __init__(self, 
                 save_path: str, 
                 style_model_path: str, 
                 batch_size: int, 
                 device: int):
        self.path = save_path
        self.batch_size = batch_size

        self.style_clf = pipeline('text-classification', model=style_model_path, top_k=None, function_to_apply='sigmoid', device=device)
        self.cola_clf = pipeline('text-classification', "cointegrated/roberta-large-cola-krishna2020", device=device)
        self.sim_scorer = compute_sim

    def get_style_reward(self, 
                         prompts: List[str], 
                         responses: List[str], 
                         epoch: str, tgt_styles: List[str] = None, 
                         eval: bool = False, 
                         src_styles = None, 
                         full_output = False, 
                         product=True,
                         style_weight = 1,
                         flu_weight = 1,
                         sim_weight = 1
        ) -> List[float]:

        reward_file = f'{self.path}/reward_{epoch}.json'
        with open(reward_file, 'w') as fo:
            for p_sents, r_sents, tgt_sts, src_sts in tqdm(batchify_zip(prompts, responses, self.batch_size, tgt_styles, src_styles), total=math.ceil(len(responses) / self.batch_size),
                              desc='Scoring generations'):
                # TODO check that inputs are still uppercased
                with torch.no_grad():
                    style_predictions = self.style_clf([r.lower() for r in r_sents])
                    acc_predictions = self.cola_clf(r_sents)
                    sim_scores = self.sim_scorer(p_sents, r_sents)

                    # TODO revisit this? should we just 0 for cleaness? (Which is what we're currently doing)
                    if product:
                        sim_scores = [max(x,0) for x in sim_scores]
                    else:
                        sim_scores = [max(x,0) for x in sim_scores]

                    orig_style_predictions = self.style_clf([p.lower() for p in p_sents])
                    orig_acc_predictions = self.cola_clf(p_sents)
            
                assert len(style_predictions) == len(tgt_sts), f"size of style predictions ({len(style_predictions)}) doesn't match size of target styles ({len(tgt_sts)})"
                
                style_prediction_probs = []
                for tgt_st, style_preds in zip(tgt_sts, style_predictions):
                    gold_label = 'LABEL_' + str(rev_clf_label_map[tgt_st])
                    for item in style_preds:
                        if item['label'] == gold_label:
                            style_prediction_probs.append(item['score'])
                            break

                style_prediction_labels = tgt_sts
                assert len(style_prediction_probs) == len(style_prediction_labels) == len(style_predictions)

                # Processing for cola fluency
                acc_prediction_probs = [self.acceptability(ex) for ex in acc_predictions]
                orig_acc_prediction_probs = [self.acceptability(ex) for ex in orig_acc_predictions]

                clean_style_predictions = [convert_label_list(s ,label_map =clf_label_map) for s in style_predictions]
                clean_orig_style_predictions = [convert_label_list(s ,label_map =clf_label_map) for s in orig_style_predictions]

                final_scores = [style_weight*i + flu_weight*j + sim_weight*k for 
                                i, j, k in zip(style_prediction_probs, acc_prediction_probs, sim_scores)]

                for fin_scr, tgt_sty, tgt_sty_prob, tgt_fl, sim_scr, src_sty, src_fl, tgt_sty_preds, src_sty_preds in zip(final_scores, style_prediction_labels, style_prediction_probs, acc_prediction_probs, sim_scores, src_sts, orig_acc_prediction_probs, clean_style_predictions, clean_orig_style_predictions):
                    fo.write(json.dumps(
                        {'score': fin_scr, 'target_style': tgt_sty, 
                        'target_style_prob': tgt_sty_prob, 'acceptability_prob': tgt_fl, 
                        'similarity': sim_scr, "source_style": src_sty, 
                        'source_acceptability_prob': src_fl, 
                        'target_all_sty_preds': tgt_sty_preds,
                        'source_all_sty_preds': src_sty_preds
                        }) + "\n")
                
        assert os.path.exists(reward_file), 'Missing reward file'
        data = pd.DataFrame.from_dict({'prompt': prompts})
        results = collate(data, responses, load_jsonl(reward_file), os.path.join(self.path, f'reward_{epoch}.json'))
        rewards = [y['score'] for x in results for y in x]
        labels = [y['target_style'] for x in results for y in x]
        
        if full_output:
            return rewards, labels, [y['target_style_prob'] for x in results for y in x], [y['acceptability_prob'] for x in results for y in x], [y['similarity'] for x in results for y in x]
        else:
            return rewards, labels
        
    def acceptability(self, result: Dict[str, Any]):
        return result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        yield {'text': generation, **response}


def collate(dataset: pd.DataFrame,
            generations: List[str],
            responses: Iterable[Dict[str, Any]],
            output_file: str = ''):
    generations_col_iter = make_generations_col(generations, responses)
    assert len(generations) % len(dataset) == 0
    n = len(generations) // len(dataset)
    generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
    dataset['generations'] = generations_col

    if output_file:
        dataset.to_json(output_file, orient='records', lines=True)
    return generations_col


if __name__ == "__main__":
    # Small example of using reward
    prompts = ["Hey, where you going"]
    responses = ["Where art thou going?"]
    batch_size = 2
    device = 0
    save_path = "temp/"
    style_model_path = "hallisky/cds_style_classifier"
    
    reward = Reward(save_path=save_path,
                    style_model_path=style_model_path, 
                    batch_size=batch_size, 
                    device=device)
    src_styles = ["aae"]
    tgt_styles = ["shakespeare"]
    reward.get_style_reward(prompts=prompts,
                            responses=responses, 
                            epoch=0, 
                            tgt_styles=tgt_styles,
                            src_styles=src_styles)