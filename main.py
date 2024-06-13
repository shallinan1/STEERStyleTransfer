import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'

import torch
import json
import logging
import random
import argparse
import numpy as np
from typing import List, Dict
from collections import namedtuple
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, pipeline
import pandas as pd
from data_pool_multitoken import DataPoolMultiToken

# Code to do external perplexity on gpt2 large
from utils.external_perplexity import get_external_perplexity
        
import time
from datetime import datetime
from arguments import get_args
from policy import Policy
from data_pool import DataPool
from reward import Reward
from utils.utils import expand_and_prepend_tokens
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, alpha_scheduler, clean_output, remove_bad_punc, load_jsonl
from utils.utils import clean_joyce, clean_coha1990, clean_coha1890, clean_coha1810, clean_lyrics, clean_shakespeare

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

from IPython import embed
from utils.constants import STYLE_MAPPING

STYLES = ["HISTORY_1", "HISTORY_2", "HISTORY_3", "BIBLE", "ENG_TWT", "LYRICS", "POETRY", "SHAKESPEARE", "SWITCHBOARD", "AAE", "JOYCE"]
assert len(STYLES) == 11
style_path = ["english_tweet", "romantic_poetry", "shakespeare", "aae", "bible", "coha_1810", "coha_1890", "coha_1990", "joyce", "lyrics", "switchboard"]

Prompt = namedtuple("Prompt", "src_sty tgt_sty prompt")

#TODO see if we want anything to do with cycling through different data
class PromptDataset(Dataset):
    def __init__(self, 
                 path, 
                 subset = 'train',
                 dataset_partition = 0,
                 seed = 1,
                 num_examples = 0,
                 include_target_only = None,
                 nonrandom_split = False,
                 override_shuffle_size = 0):
        data, self.prompt = [], []
        
        # Format (style_type, prompt)
        for sty in style_path:
            with open(os.path.join(path, sty, f"tmp/{subset}.txt"), 'r') as f:
                cur = [s.strip() for s in f.readlines()]
                
                if num_examples != 0:
                    if nonrandom_split:
                        cur = cur[(num_examples * dataset_partition):(num_examples * (dataset_partition + 1))]
                    else:
                        # Make a new random instance
                        data_random = random.Random(seed)

                        new_cur = None
                        if override_shuffle_size > 0:
                            for _ in range(dataset_partition+1):
                                data_random.shuffle(cur)
                                new_cur = cur[:override_shuffle_size]
                                cur = cur[override_shuffle_size:]
                            cur = new_cur[:num_examples]

                        else:
                            for _ in range(dataset_partition+1):
                                data_random.shuffle(cur)
                                new_cur = cur[:num_examples]
                                cur = cur[num_examples:]
                            cur = new_cur

                    assert len(cur) == num_examples
                
                cur = [clean_output(c) for c in cur]
                # Style specific cleaning here to match train multilabel
                if sty == "joyce":
                    cur = [clean_joyce(t) for t in cur]
                elif sty == "coha_1990":
                    cur = [clean_coha1990(t) for t in cur]
                elif sty == "coha_1890":
                    cur = [clean_coha1890(t) for t in cur]
                elif sty == "coha_1810":
                    cur = [clean_coha1810(t) for t in cur]
                elif sty == "lyrics":
                    cur = [clean_lyrics(t) for t in cur]
                elif sty == "shakespeare":
                    cur = [clean_shakespeare(t) for t in cur]

                cur = [remove_bad_punc(c) for c in cur]
                data.extend([(sty, s) for s in cur])

                print("Loaded style", sty, "with", str(len(cur)), "examples")
        for example in data:
            if not include_target_only:
                # Expand each input in style x with all other styles != x
                self.prompt.extend([Prompt(example[0], st, example[1]) for st in [j for j in style_path if j != example[0]]])
            if include_target_only:
                self.prompt.extend([Prompt(example[0], st, example[1]) for st in [j for j in [include_target_only] if j != example[0]]])

        # Sort prompts according to styles to have same styles in each batch for sampling each using style lm. 
        self.prompt = sorted(self.prompt, key = lambda x: (x.tgt_sty, x.src_sty))
    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, idx):
        return {
            'src_style': self.prompt[idx].src_sty,
            'tgt_style': self.prompt[idx].tgt_sty,
            'prompt': self.prompt[idx].prompt
        }

class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

    def __call__(self, sequences):
        prompts = [sequence['prompt'] for sequence in sequences]
        src_styles = [sequence['src_style'] for sequence in sequences]
        tgt_styles = [sequence['tgt_style'] for sequence in sequences]

        # Limit input sequences to 50 tokens, then add a <bos> token for paraphrasing
        prompts = [self.tokenizer.decode(self.tokenizer.encode(p)[:50]).strip() + " <bos>" for p in prompts]
        # Make max length 51 - sequence (50 tokens) + bos token (1 token)
        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True, max_length=51, truncation=True)
        input_ids, attention_mask = encodings_dict.input_ids, encodings_dict.attention_mask

        return input_ids, attention_mask, src_styles, tgt_styles


class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool):
        self.queries, self.responses, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx]
                }


class SequenceCollator(object):
    def __init__(self, tokenizer, num_reward_tokens):
        self.tokenizer = tokenizer
        self.num_reward_tokens = num_reward_tokens
        self.tokenizer.padding_side = "left"

    def __call__(self, sequences):
        # Technically don't need to do truncation to length of 50 here because we did that in the preprocessing
        queries = [self.tokenizer.decode(self.tokenizer.encode(sequence['query'])[:50]).strip() + " <bos>" for sequence in sequences]
        responses = [self.tokenizer.decode(self.tokenizer.encode(sequence['response'].strip())[:50]).strip()  + " <eos>" for sequence in sequences]
        combos = [q + " " + r for q,r in zip(queries, responses)]

        encodings_dict = self.tokenizer(combos, return_tensors="pt", padding=True, max_length=102, truncation=True)
        input_ids, attention_mask = encodings_dict['input_ids'], encodings_dict['attention_mask']

        # Need to check if cat tokens is a list or not
        cat_tokens = [sequence['cat_tokens'] for sequence in sequences]
        if isinstance(cat_tokens[0], list):
            cat_ids = [[self.tokenizer.convert_tokens_to_ids(cat_tok) for cat_tok in cur_cat] for cur_cat in cat_tokens]
        else:
            cat_ids = [[self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens'])] for sequence in sequences]

        input_ids, attention_mask = expand_and_prepend_tokens(input_ids, attention_mask, cat_ids)

        print("SEQ sequence collator")
        assert (input_ids == self.tokenizer.bos_token_id).any(dim=1).all()
        assert (input_ids == self.tokenizer.eos_token_id).any(dim=1).all()
        return input_ids, attention_mask


class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult

class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: Policy,
                 style_lm_dict: Dict,
                 data_pool: DataPool,
                 score_model: Reward,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 precomputed_df = None,
                 use_experts = False,
                 multiple_reward_tokens=False,
                 style_weight = 1.0,
                 sim_weight = 1.0,
                 flu_weight = 1.0
                 ):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.expert_style = style_lm_dict
        self.data_pool = data_pool
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # self.writer = SummaryWriter(comment=f"target:{self.params.target_sentiment}")
        self.writer = SummaryWriter(self.params.tensorboard_dir)
        self.precomputed_df = precomputed_df
        self.use_experts = use_experts
        self.style_weight = style_weight
        self.sim_weight = sim_weight
        self.flu_weight = flu_weight

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.multiple_reward_tokens = multiple_reward_tokens
        self.num_reward_tokens = 3 if self.multiple_reward_tokens else 1
        print(str(self.num_reward_tokens) + " NUM REWARD TOKENS")
        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer, num_reward_tokens = self.num_reward_tokens)

        # If we load from checkpoint, we need to initialize this
        if len(self.data_pool.datapool_data) != 0:
            sample_dataset = SequenceDataset(data_pool=self.data_pool)
            self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                                shuffle=True, drop_last=False, collate_fn=self.seq_collator)
            self.sampler = iter(self.sample_dataloader)

    # TODO this doesn't work if we add multiple target styles??
    def add_control_code(self, input_ids, attention_mask, target_styles):
        if not self.multiple_reward_tokens:
            # TODO potentially change this logic to make it more generalizable. currently, highest bin is best
            best_cat_ids = [
                [self.policy.tokenizer.convert_tokens_to_ids(
                    self.tree_tokens[STYLE_MAPPING[target_style]][-1])] for target_style in target_styles
                    ]

            input_ids, attention_mask = expand_and_prepend_tokens(input_ids, attention_mask, best_cat_ids, self.policy.tokenizer.pad_token_id)
        else:
            # Multiple reward tokens. Order: style, similarity, fluency
            best_cat_ids = [[a,b,c] for a,b,c in zip(
                [self.policy.tokenizer.convert_tokens_to_ids(self.tree_tokens["style"][STYLE_MAPPING[target_style]][-1]) for target_style in target_styles],
                [self.policy.tokenizer.convert_tokens_to_ids(self.tree_tokens["similarity"][STYLE_MAPPING[target_style]][-1]) for target_style in target_styles],
                [self.policy.tokenizer.convert_tokens_to_ids(self.tree_tokens["fluency"][STYLE_MAPPING[target_style]][-1]) for target_style in target_styles],
            )]
            input_ids, attention_mask = expand_and_prepend_tokens(input_ids, attention_mask, best_cat_ids, self.policy.tokenizer.pad_token_id)

        return input_ids, attention_mask

    def decode(self, query_input_ids, response_input_ids=None):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for p in query_input_ids]

        if response_input_ids is None:
            return query

        response = [self.policy.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for r in response_input_ids]
        return query, response

    def sample(self, step):
        if step % self.params.sample_interval != 0:
            return
        log.info(f"[step {step}] Sampling ...")

        prompts, responses, orig_styles, target_styles = [], [], [], []

        if step == 0 and self.precomputed_df is not None:
            log.info(f"Using precomputed df!")

            prompts, responses, orig_styles = self.precomputed_df.prompt.tolist(), self.precomputed_df.text.tolist(), self.precomputed_df.source_style.tolist()
            if self.multiple_reward_tokens:
                scores, labels, style_scores, fluency_scores, similarity_scores = self.precomputed_df.score.tolist(), self.precomputed_df.target_style.tolist(), \
                    self.precomputed_df.target_style_prob, self.precomputed_df.acceptability_prob, self.precomputed_df.similarity
            else:
                scores, labels = self.precomputed_df.score.tolist(), self.precomputed_df.target_style.tolist()
        else:
            # TODO check to make sure the left padding is correct
            for _, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader),
                                        desc='Sampling from current policy')):
                input_ids, attention_mask, src_styles, tgt_styles = batch

                # Faeze: we sample from expert style lms
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask, tgt_styles)
                if self.use_experts:
                    style_lm = self.expert_style[tgt_styles[0]]
                    rollouts = self.policy.sample_experts(style_lm, input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p,
                                    filter_p=self.params.filter_p, alpha=alpha_scheduler(step, self.params.alpha), tree_tokens_added=True, no_repeat_ngrams=self.params.no_repeat_ngrams)
                else:
                    #TODO min len?
                    rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p,
                                    max_len = self.params.max_gen_length, min_len = 2, sample=True, no_repeat_ngrams=self.params.no_repeat_ngrams)

                response = rollouts['response/text']
                prompt = self.decode(rollouts['query/input_ids'][:, self.num_reward_tokens:])
                prompts.extend(prompt)
                responses.extend(response)
                orig_styles.extend(src_styles)
                target_styles.extend(tgt_styles)

            if self.multiple_reward_tokens:
                scores, labels, style_scores, fluency_scores, similarity_scores = self.score_model.get_style_reward(
                    prompts, responses, f'step{step}', tgt_styles=target_styles, src_styles=orig_styles, full_output=True, 
                    style_weight=self.style_weight, flu_weight=self.flu_weight, sim_weight=self.sim_weight)
            else:
                scores, labels = self.score_model.get_style_reward(prompts, responses, f'step{step}', tgt_styles=target_styles, src_styles=orig_styles, style_weight=self.style_weight, flu_weight=self.flu_weight, sim_weight=self.sim_weight)
            
            assert target_styles == labels # Sanity check

        if self.multiple_reward_tokens:
            self.data_pool.add_style(prompts=prompts, responses=responses, scores=scores, styles=labels, orig_styles=orig_styles, \
                                     style_scores=style_scores, similarity_scores=similarity_scores, fluency_scores=fluency_scores)
        else:
            self.data_pool.add_style(prompts=prompts, responses=responses, scores=scores, styles=labels, orig_styles=orig_styles)
        # TODO expand this so that it is not just on score here
        self.writer.add_histogram('Train/multi-style-score-hist-all', self.data_pool.datapool_data.score.to_numpy(), step)
        self.writer.add_scalar('Train/multi-style-score-all', self.data_pool.datapool_data.score.mean(), step)
        for style, temp_df in self.data_pool.datapool_data.groupby('style'):
            self.writer.add_histogram(f'Train/multi-style-score-hist-{style}', temp_df.score.to_numpy(), step)
            self.writer.add_scalar(f'Train/multi-style-score-{style}', temp_df.score.mean(), step)

        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, drop_last=False, collate_fn=self.seq_collator)
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):
        step_started_at = time.time()
        self.sample(step=step_num)

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)

        self.optimizer.zero_grad()
        ppo_loss, stats = self.loss(step_num, *batch)
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        for metric in ['kl', 'entropy']:
            self.writer.add_scalar(f'Objective/{metric}', stats[f'objective/{metric}'], step_num)
        for metric in ['lm', 'kl', 'entropy', 'total']:
            self.writer.add_scalar(f'Loss/{metric}', stats[f'loss/{metric}'], step_num)
        self.writer.add_scalar(f'Params/lr', self.optimizer.param_groups[0]['lr'], step_num)
        self.writer.add_scalar(f'Params/kl_coef', self.kl_ctl.value, step_num)
        self.writer.add_scalar(f'Params/entropy_coef', self.entropy_ctl.value, step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True)
        self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        log.info(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        self.save(step=step_num)
        self.eval(step=step_num)


    def loss(self, step, query_input_ids, query_mask, response_input_ids, response_mask):
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        if self.multiple_reward_tokens:
            total_len = 0
            for key in self.tree_tokens:
                total_len += len([y for x in list(self.tree_tokens[key].values()) for y in x])
            # print(total_len)
            logits = outputs['response/logits'][:, :, :-total_len]
        else:
            logits = outputs['response/logits'][:, :, :-len(self.tree_tokens) * self.params.n_extra_tokens]
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            ref_outputs = self.ref_policy.forward_pass(query_input_ids[:, self.num_reward_tokens:], query_mask[:, self.num_reward_tokens:],
                                                       response_input_ids, response_mask)
            ref_logprobs, ref_logits = ref_outputs['response/log_prob'], ref_outputs['response/logits']

        kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)

        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = self.decode(query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step)

        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(4, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            self.writer.add_text('rewrite_examples', f'{queries[i]} ==> {responses[i]}', global_step=step)
            print("\nquery: " + queries[i] + "\nresponse: " + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save(self, step):
        # Don't save initial model
        if step == 0:
            return
        if step % self.params.save_interval != 0:
            return
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'data_pool': self.data_pool.datapool_data 
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        log.info(f"[step {step}] model checkpoint saved")

    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[step {step}] evaluating ...")

        self.policy.model.eval()
        # Seems to not be doing well when it doesn't have punctuation...........
        print("\n\n\n-------\n")
        sents = ["Martin was a smart man, and above all, my friend <bos>",
                 "Martin was a smart man, and above all, my friend. <bos>",
                 "Dogs are fast, loyal, and good at eating food! <bos>",
                 "I saw Mabellene in a Coup de Ville <bos>",
                 "My belongings scattered across the hotel floor <bos>",
                 "Which way ran he that kill'd Mercutio? <bos>",
                 "We will go to the tower to eat a hearty lunch. <bos>",
                 "Woe to her that is filthy and polluted, to the oppressing city! <bos>",
        ]
        for s in sents:
            print(self.policy.tokenizer.batch_decode(self.policy.model.generate(self.policy.tokenizer(s, return_tensors="pt").input_ids.to('cuda'), max_length=50, eos_token_id=self.policy.tokenizer.eos_token_id, num_return_sequences=1, do_sample=False)),"\n")

        # print("NOW PRINTING WITH CONTROL")
        # # Try printing with the control token
        # for s in sents:
        #     tokenized = self.policy.tokenizer(s, return_tensors="pt")
        #     input_ids, _ = self.add_control_code(tokenized["input_ids"], tokenized["attention_mask"], ["romantic_poetry"])
        #     print(self.policy.tokenizer.batch_decode(self.policy.model.generate(input_ids.to('cuda'), max_length=50, eos_token_id=self.policy.tokenizer.eos_token_id, do_sample=False)),"\n")

        # print("DONE WITH CONTROL")
        # TODO move eval?

        generations, all_prompts, perplexities, tgt_labels, src_labels = [], [], [], [], []
        for i, (input_ids, attention_mask, src_styles, tgt_styles) in enumerate(tqdm(self.val_dataloader, desc="Eval loop")):
            target_styles = tgt_styles
            with torch.no_grad():
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask, target_styles)
                
                # TODO change sample since the outputs look weird
                rollouts = self.policy.sample2(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    top_p=self.params.top_p,
                    max_len=self.params.max_gen_length, 
                    min_len=2, 
                    sample=True, 
                    no_repeat_ngrams=self.params.no_repeat_ngrams)   
                
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'][:, self.num_reward_tokens:],
                                  'query_mask': rollouts['query/mask'][:, self.num_reward_tokens:],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask']}
                ref_policy_outputs = self.ref_policy.forward_pass(**forward_inputs)
                ref_logprobs, ref_loss = ref_policy_outputs['response/log_prob'], ref_policy_outputs['response/lm_loss']
                perplexity = torch.exp(reduce_mean(ref_loss, rollouts['response/mask'].float(), axis=1))
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                prompt = self.decode(rollouts['query/input_ids'][:, self.num_reward_tokens:])
                response = rollouts['response/text']

                generations.extend(response)
                all_prompts.extend(prompt)
                tgt_labels.extend(tgt_styles)
                src_labels.extend(src_styles)

        # TODO: modify scoring func for eveluation: here we care style correctness for the target style
        scores, labels, target_style_probs, acceptability_probs, similarities = self.score_model.get_style_reward(all_prompts, generations, f'step{step}_eval{i}', tgt_styles=tgt_labels, eval=True, src_styles=src_labels, full_output=True,style_weight=self.style_weight, flu_weight=self.flu_weight, sim_weight=self.sim_weight)
        assert labels == tgt_labels

        eval_df = pd.DataFrame({"score": scores, "label": labels, "target_style_prob": target_style_probs, "acceptability_prob": acceptability_probs, "similarity": similarities})
        for style, row in eval_df.groupby("label").mean().iterrows():
            for col in eval_df.columns:
                if col == "label":
                    continue
                self.writer.add_scalar('EvaluationStyleSpecific/'+style + "_" + col, row[col], step)

        print("Getting external perplexity on gpt2xl")
        external_perplexity = [min(1e4, a) for a in get_external_perplexity(generations)]

        ppl_score, style_score = np.mean(perplexities), np.mean(scores)
        print(f"  external perplexity = {np.nanmean(external_perplexity):+.2f}")
        print(f"  perplexity = {ppl_score:+.2f}")
        print(f"  style scores (agg.) = {style_score:+.4f}")
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)
        self.writer.add_scalar('Evaluation/perplexity_gpt2xl', np.nanmean(external_perplexity), step)

        self.writer.add_scalar('Evaluation/multi-style-score', style_score, step)
        self.writer.add_scalar('Evaluation/target-style-prob', np.mean(target_style_probs), step)
        self.writer.add_scalar('Evaluation/acceptability-prob', np.mean(acceptability_probs), step)
        self.writer.add_scalar('Evaluation/similarity', np.mean(similarities), step)


def main():
    start_time = time.time()
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Make directories to save checkpoints, tensorboard, etc. Do not remake if we are loading from checkpoint
    if not args.load_from_ckpt:
        cur_time = datetime.now()
        date_time = cur_time.strftime("%m-%d-%Y_%H:%M:%S")
        args.save_dir = os.path.join(f'{args.output_dir}', date_time)    
        args.tensorboard_dir = os.path.join("runs/", date_time)
        if args.save_naming:
            args.save_dir += "_" + args.save_naming
            args.tensorboard_dir += "_" + args.save_naming
        args.reward_dir = os.path.join(args.save_dir, 'reward')
        args.model_dir = os.path.join(args.save_dir, 'model')

        print(args.tensorboard_dir)

        for d in [args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
            ensure_dir(d)
        
        log.info(f'Write to output directory: {args.save_dir}')

        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    #TODO add argument to do different tokens for different rewards
    #TODO try disentangling the reward tokens from the style tokens
    # TODO support differnet number of buckets for different reward tokens
    if args.multiple_reward_tokens:
        tree_tokens = {}
        for metric in ["style", "similarity", "fluency"]:
            tree_tokens[metric] = {st: [' _{}_{}_'.format(st, str(idx).zfill(5)) + metric for idx in range(args.n_extra_tokens)] 
                for st in STYLES
                }
        data_pool = DataPoolMultiToken(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens)
    else:
        tree_tokens = {st: [' _{}_{}'.format(st, str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] 
            for st in STYLES
            }
        data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens)

    log.info(f'Initializing quark models ...')
    ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device)
    policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device,
                    reward_cond=True, tree_tokens=tree_tokens)

    # Initilaize optimizer and scheduler
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-5)
    print("total steps", args.total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)
    
    if args.load_from_ckpt is not None:
        print("\t LOADING FROM CHECKPOINT")
        checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
        policy.model.load_state_dict(checkpoint['policy_model'])

        # Load data pool, optimizer, scheduler
        data_pool.datapool_data = checkpoint['data_pool'] 
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        checkpoint.clear()
        assert policy.model.training == True 
        print("\t Finished loading!")

        args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
        args.tensorboard_dir = args.save_dir.replace("outputs", "runs")
        args.reward_dir = os.path.join(args.save_dir, 'reward')
        args.model_dir = os.path.join(args.save_dir, 'model')

        print(args.tensorboard_dir)

    reward = Reward(save_path=args.reward_dir, style_model_path=args.reward_model_dir, 
                    batch_size=args.batch_size, device=num_gpus-1)
        
    # TODO: remove these tokens from intial policy IF we do non paraphraser initial policy + using experts
    # added_tokens = ['<dense-vectors>', '<tokens>', '<verb>', '<ARG0>', '<ARG1>', '<global-dense-vectors>', '<pad>', '<bos>', '<eos>'] + [y for x in list(tree_tokens.values()) for y in x]

    style_lm_dict = None
    # By default, DON'T use this
    if args.use_experts:
        # Load expert style lms
        style_lm_dict = dict.fromkeys(style_path)
        log.info(f'Initializing expert style models ...')

        # Adding extra tokens in the pre-trained diverse paraphraser
        added_tokens = [y for x in list(tree_tokens.values()) for y in x]
        for sty in style_path:
            model_path = os.path.join(args.expert_dir, sty)
            style_lm_dict[sty] = Policy(model_name=model_path, temperature=args.temperature, device=num_gpus - 2, tree_tokens=added_tokens)
    else:
        log.info(f'\nNOT initializing expert style models ...')

    log.info(f'Initialization done!')

    precomputed_df = None

    weight_total = args.sim_weight + args.flu_weight + args.style_weight
    args.sim_weight /= weight_total
    args.flu_weight /= weight_total
    args.style_weight /= weight_total

    # Load in the pre-computed data
    if args.load_from_ckpt is None:
        log.info(f'Loaded pre-computed data!')
        precomputed_data = [x for x in load_jsonl(os.path.join(args.precomputed_dataset_dir))]
        precomputed_df = pd.DataFrame(precomputed_data)
        # Make sure that we shuffle the data
        precomputed_df = precomputed_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        print("Will take at least", str(len(precomputed_df) // args.batch_size), " steps to go through initial data")
        print("Will take at least", str((args.num_examples_train * 11 * 10)// args.batch_size), " steps to go through training data")
        log.info(f'Loading pre-computed data done!')

        precomputed_df.score = args.sim_weight * precomputed_df.similarity \
            + args.style_weight * precomputed_df.target_style_prob \
                + args.flu_weight * precomputed_df.acceptability_prob
        log.info("Re-weighting done!")

        # pipeline_texts_forward = [f"premise: {a} hypothesis: {b}" for a, b in zip(precomputed_df["prompt"], precomputed_df["text"])]
        # pipeline_texts_reverse = [f"premise: {a} hypothesis: {b}" for a, b in zip(precomputed_df["text"], precomputed_df["prompt"])]

        # pipe = pipeline("text2text-generation", model="google/t5_xxl_true_nli_mixture", device=2)
        # pipe(pipeline_texts_forward)

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)

    #TODO: experiment with whether we want to include prompts seen in the precompute or not
    train_dataset = PromptDataset(
        path=args.dataset_dir, subset='train', num_examples=args.num_examples_train,
        nonrandom_split=args.nonrandom_split, seed=args.seed, dataset_partition=args.dataset_partition, override_shuffle_size = 1000
        )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(
        path=args.dataset_dir, subset='dev', num_examples=args.num_examples_val, 
        nonrandom_split=args.nonrandom_split, seed=args.seed)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    trainer = ConditionTrainer(params=args, 
                               policy=policy, 
                               ref_policy=ref_policy, 
                               style_lm_dict=style_lm_dict, 
                               data_pool=data_pool,
                               score_model=reward, 
                               tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, 
                               val_dataloader=val_dataloader,
                               optimizer=optimizer, 
                               scheduler=scheduler, 
                               precomputed_df=precomputed_df, 
                               use_experts=args.use_experts, 
                               multiple_reward_tokens=args.multiple_reward_tokens,
                               style_weight=args.style_weight,
                               sim_weight=args.sim_weight,
                               flu_weight=args.flu_weight)
    
    if not args.load_from_ckpt:
        step_num = 0
    else:
        step_num = int(args.load_from_ckpt.rsplit("_",1)[1][:-4]) + 1
    print(f"Starting from step {step_num}")

    while step_num < args.total_steps:
        try:
            trainer.step(step_num)
            step_num += 1
        except RuntimeError as e:
            print(e)
            torch.cuda.empty_cache()
            continue

    print("\n Program finished in", str(round(time.time()-start_time, 2)), "seconds")

# TODO visualize data after preprocessing
# TODO try pretraining more on just style, not control tokens
if __name__ == "__main__":
    main()

"""
Todo: try different dataset, different bins, different top p


# For RTX6k
bs 128, 256

# For A100, 4 GPUS
python3 main.py \
    --precomputed_dataset_dir training_data_topk/top200k_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 256 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_5e-4_p0.9_200k_bs256
    
# For A100, 4 GPUS
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent_5buckets/min0_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 256 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 250 \
    --eval_interval 250 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_1e-4_p0.9_uniform100min0_bs256_multi \
    --multiple_reward_tokens

# Batch size 324, multi reward
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_50percent_5buckets_min5k/min0.001_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 324 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 250 \
    --eval_interval 250 \
    --kl_coef 0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --multiple_reward_tokens \
    --save_naming product_1e-4_p0.9_half_min5k_bs324_multireward  
    
# Batch size 128, multi reward
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_50percent_5buckets_min5k/min0.001_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 128 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 1e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.05 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --multiple_reward_tokens \
    --save_naming product_1e-4_p0.9_half_min5k_bs128_multireward  

# Batch size 128, multi reward, kl coef
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_50percent_5buckets_min5k/min0.001_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 128 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 1e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --multiple_reward_tokens \
    --save_naming product_1e-4_p0.9_half_min5k_bs128_multireward  

# Batch size 128 with min 5k dataset
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent_5buckets_min20k/min0_balanced_data_combo.jsonl\
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 128 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 1e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_1e-4_p0.9_fullmin20k_bs128 \


# Batch size 128
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent/min0_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 128 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 1e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_1e-4_p0.9_bigdataset_bs128 \

# Batch size 64, higher learning rate
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent_5buckets_min20k/min0_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 64 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 10000000 \
    --lr 1e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 5000 \
    --save_interval 1000 \
    --eval_interval 1000 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_1e-4_p0.9_fullmin20k_bs64 \


# Batch size 64, higher learning rate
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent/min0_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 64 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 10000000 \
    --lr 2e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 30 \
    --top_p 0.9 \
    --sample_interval 5000 \
    --save_interval 1000 \
    --eval_interval 1000 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_2e-4_p0.9_hugedataset_bs64 \


###############
    
# CONTINUE training from checkpoint command
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent_5buckets_min20k/min0.1_balanced_data_combo.jsonl \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 128 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming 15.5k_RESUME \
    --load_from_ckpt outputs/04-18-2023_20:00:52_product_5e-4_p0.9_uniform100min0.1min20k_bs128_multi/model/ckp_17000.pth \
    --multiple_reward_tokens


########


# CONTINUE training from checkpoint command number 2 (batch size 256)
python3 main.py \
    --precomputed_dataset_dir training_data/uniform_100percent_5buckets_min1k/min0.05_balanced_data_combo.jsonl\
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 256 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming RESUME \
    --load_from_ckpt outputs/04-17-2023_06:37:11_product_5e-4_p0.9_uniform100min0.1min20k_bs256_multi/model/ckp_20000.pth

    
    #####
    For Mosaic runnning

    # For A6000 GPUS, with min 5k and min 0.025
    python3 main.py \
        --precomputed_dataset_dir training_data/uniform_100percent_5buckets_min20k/min0.1_balanced_data_combo.jsonl \
        --n_extra_tokens 5 \
        --init_model models/paraphraser_gpt2_large \
        --ref_model models/paraphraser_gpt2_large \
        --temperature 1.0 \
        --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
        --batch_size 256 \
        --dataset_partition 10 \
        --num_examples_val 100 \
        --num_examples_train 1000 \
        --total_episodes 20000000 \
        --lr 1e-4 \
        --num_warmup_steps 500 \
        --max_gen_length 50 \
        --top_p 0.9 \
        --sample_interval 2500 \
        --save_interval 500 \
        --eval_interval 500 \
        --kl_coef 0.0 \
        --entropy_coef 0.0 \
        --no_repeat_ngrams 3 \
        --save_naming product_1e-4_p0.9_uniform100min0.1min20k_bs256_multi \
        --multiple_reward_tokens \
        --dataset_dir datasets/cds \
    

    # Try out the old data
    python3 main.py \
    --precomputed_dataset_dir training_data/OLD/uniform_100percent_5buckets_min20k/min0_balanced_data_combo.jsonl  \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 128 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.95 \
    --sample_interval 2500 \
    --save_interval 500 \
    --eval_interval 500 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 3 \
    --save_naming product_5e-4_p0.9_OLDuniform100min0.05min1k_bs128
 
### 



TRY USING 

# For a100, 200k
python3 main.py \
    --precomputed_dataset_dir training_data_topk/top200k_data_combo.jsonl  \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 300 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 100 \
    --eval_interval 100 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 0 \
    --multiple_reward_tokens \
    --save_naming product_5e-4_p0.9_800k_bs128_multi \
    

# For a100, 200k SMALL VERISON
python3 main.py \
    --precomputed_dataset_dir training_data_topk/top200k_data_combo.jsonl  \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 300 \
    --dataset_partition 10 \
    --num_examples_val 1 \
    --num_examples_train 10 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 10 \
    --eval_interval 10 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 0 \
    --multiple_reward_tokens \
    --save_naming product_5e-4_p0.9_800k_bs128_multi \
    

## Weight the other rewards more
# For a100, 200k
python3 main.py \
    --precomputed_dataset_dir training_data_topk/top200k_data_combo.jsonl  \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 300 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 100 \
    --eval_interval 100 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 0 \
    --save_naming product_5e-4_p0.9_800k_bs128_multi \
    --sim_weight 2.0 \
    --style_weight 1.0 \
    --flu_weight 1.0

# For a100, 200k
python3 main.py \
    --precomputed_dataset_dir hallisky/STEER-data-top-400k-combo  \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir models/multilabel/03-21-2023_21:54:15/checkpoint-4272 \
    --batch_size 300 \
    --dataset_partition 10 \
    --num_examples_val 100 \
    --num_examples_train 1000 \
    --total_episodes 20000000 \
    --lr 5e-4 \
    --num_warmup_steps 500 \
    --max_gen_length 50 \
    --top_p 0.9 \
    --sample_interval 2500 \
    --save_interval 100 \
    --eval_interval 100 \
    --kl_coef 0.0 \
    --entropy_coef 0.0 \
    --no_repeat_ngrams 0 \
    --multiple_reward_tokens \
    --save_naming product_5e-4_p0.9_400k_bs128_multi \

"""