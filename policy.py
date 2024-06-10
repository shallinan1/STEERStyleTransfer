import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.constants import NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad, clean_output
from utils.generation_utils import top_k_top_p_filtering
from logits_process import NoRepeatNGramLogitsProcessor
from IPython import embed

class Policy:
    def __init__(self, 
                 model_name: str, 
                 temperature: float, 
                 device: int, 
                 reward_cond: bool = False, 
                 tree_tokens = None):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = 'left' # Needed for batch generation
        self.tree_token_ids = None

        if reward_cond or tree_tokens:
            print("\nADDING TOKENS:", tree_tokens)
            if isinstance(tree_tokens, Dict):
                # Hacky check to see if we're using multi tokens or not
                if "style" in tree_tokens:
                    tree_tokens = [y for x in list(tree_tokens["style"].values()) for y in x] + [y for x in list(tree_tokens["similarity"].values()) for y in x] + \
                        [y for x in list(tree_tokens["fluency"].values()) for y in x]
                else:
                    tree_tokens = [y for x in list(tree_tokens.values()) for y in x]
            tokens_added = self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

            # Need to check all the tokens were new
            assert(tokens_added) == len(tree_tokens)

            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens])

            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-len(tree_tokens):, :] = new_inits

            self.tree_token_ids = self.tokenizer.convert_tokens_to_ids(tree_tokens)

        # Manually make the device map
        device_map = None
        if torch.cuda.device_count() == 8:
            device_map = {
                0: [0],
                1: [1,2,3],
                2: [4,5,6,7,8,9],
                3: [10,11,12,13,14,15],
                4: [16,17,18,19,20],
                5: [21,22,23,24,25,26],
                6: [27,28,29,30,31],
                7: [32,33,34,35]
            }
        elif torch.cuda.device_count() == 7:
            device_map = {
                0: [0],
                1: [1,2,3,4,5],
                2: [6,7,8,9,10,11],
                3: [12,13,14,15,16,17],
                4: [18,19,20,21,22,23],
                5: [24,25,26,27,28,29],
                6: [30,31,32,33,34,35]
            }
        elif torch.cuda.device_count() == 6:
            device_map = {
                0: [0],
                1: [1,2,3,4,5,6,7],
                2: [8,9,10,11,12,13,14,15],
                3: [16,17,18,19,20,21,22],
                4: [23,24,25,26,27,28,29],
                5: [30,31,32,33,34,35]
            }
        if torch.cuda.device_count() == 4 and "A100" in torch.cuda.get_device_name():
            device_map = {
                0: [0, 1],
                1: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                2: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                3: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
            }
            
        self.model = self.model.to(self.device)
        self.model.train()
        self.model.parallelize(device_map=device_map)
        self.temperature = temperature

    # TODO check if we want to retain eos/bos token in prompts/response_text
    # Note: setting no_repeat_ngrams > 0 often makes the generations worse.
    def sample2(self,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 20,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               no_repeat_ngrams: int = 0) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        # Save the prompts
        prompts =  self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        self.model.eval()
        if temperature is None:
            temperature = self.temperature

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        input_seq_len = input_ids.shape[1]

        # This is working (verified from above )
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=3,
            do_sample=False,
            min_length=min_len,
            max_length=max_len+input_seq_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=(no_repeat_ngrams if no_repeat_ngrams > 0 else None),
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response_ids = outputs[:, input_seq_len:]
        response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # Add postprocessing
        response_text_cleaned = [clean_output(out) for out in response_text]

        # for x, (s, o) in enumerate(zip(single_outputs, response_text)):
        #     if s != o:
        #         print(s, "\n", o, "\n",prompts[x], x, "\n\n")
        self.model.train()

        return {
                'query/input_ids': input_ids,
                'query/text': prompts,
                'query/mask': attention_mask,
                'response/input_ids': response_ids,
                'response/text': response_text_cleaned,
                'response/mask': (response_ids != self.tokenizer.pad_token_id).long(),
            }

    def sample_experts(self,
               style_expert = None, 
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 30,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = 0,
               top_p: float = 1.0,
               temperature: float = 1.0,
               filter_p: float = None,
               alpha: float = 0.0,
               tree_tokens_added: bool = False,
               style_antiexpert = None,
               no_repeat_ngrams = 0,
               ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        no_repeat_ngrams_processor = None
        if no_repeat_ngrams != 0:
            no_repeat_ngrams_processor = NoRepeatNGramLogitsProcessor(no_repeat_ngrams)

        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1

        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)
        expert_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.tokenizer.encode("<eos>")[0]
        expert_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)

        # since it's style transfer task, src and tgt are likely to be the same size
        # max_len = max(max_len, input_ids.shape[1])

        self.model.eval()
        if style_expert:
            style_expert.model.eval()

        if style_antiexpert:
            style_antiexpert.model.eval()

        with torch.no_grad():
            for step in range(max_len):
                # base model prediction
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids,) # **model_kwargs)
                base_logits = outputs.logits
                
                # expert prediction
                if style_expert:
                    expert_logits = style_expert.model(expert_ids, attention_mask=expert_attention_mask).logits
                    
                else:
                    alpha = 0
                    expert_logits = base_logits

                # antiexpert prediction
                if style_antiexpert:
                    antiexpert_logits = style_antiexpert.model(expert_ids, attention_mask=expert_attention_mask).logits

                # No repeat ngram processor
                # Hack: we don't need to calculate no repeat ngrams for step 0
                if no_repeat_ngrams_processor and step != 0:
                    base_logits[:, -1, :] = no_repeat_ngrams_processor(expert_ids, base_logits[:, -1, :])

                if filter_p < 1.0:
                    base_logits = top_k_top_p_filtering(base_logits, top_p=filter_p)
                
                # DExperts
                ensemble_logits = base_logits
                
                # in the first decoding step, we want to use the 'real' last position for each sentence

                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :] # (bs, voc)
                    if style_antiexpert:
                        next_token_logits += alpha * (expert_logits[:, -1, :] - antiexpert_logits[:, -1, :])
                    # Buggy branch
                    else:
                        next_token_logits += alpha * (expert_logits[:, -1, :]) #- antiexpert_logits[range(batch_size), last_non_masked_idx, :])
                else:
                    if style_antiexpert:
                        next_token_logits = ensemble_logits[:, -1, :] + alpha * (expert_logits[:, -1, :] - antiexpert_logits[:, -1, :])
                    else:
                        next_token_logits = ensemble_logits[:, -1, :] + alpha * (expert_logits[:, -1, :])

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if top_k > 0 or top_p < 1.0:
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                    # TODO figure out why we need to add this...
                    next_token_logits[next_token_logits.isnan()] = float('-Inf')
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1) # [bs, voc]
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) # [bs]

                else:
                    next_token_logits[next_token_logits.isnan()] = float('-Inf')
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <BOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents) # [bs]

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id # or tokens_to_add == 198
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

                # update expert ids, attention mask
                expert_ids = torch.cat([expert_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                expert_attention_mask = torch.cat([expert_attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)

        
        # embed()
        response_ids = input_ids[:, input_seq_len:]
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]
        # clean outputs: remove \n newlines, etc.
        response_text = [clean_output(out) for out in response_text]

        prompt_ids = input_ids[:, :input_seq_len]
        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for query in prompt_ids]

        return {
            'query/input_ids': prompt_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
        }
        
    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor):

        # TODO might need to rewrite this based on the padding
        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)
        # embed()

        batch_size, query_seq_len = query_input_ids.shape
        input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
        model_kwargs = {'attention_mask': torch.cat([query_mask, response_mask], dim=-1)}
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        try:
            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        except:
            embed()
        # get the first logit
        query_logits = outputs.logits[:, :query_seq_len, :]
        last_non_masked_idx = torch.sum(query_mask, dim=1) - 1
        first_logits = query_logits[range(batch_size), last_non_masked_idx, :]
        # get the second to last logit
        response_logits = outputs.logits[:, query_seq_len:-1, :]
        logits = torch.cat([first_logits[:, None], response_logits], dim=1)

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        lm_loss = -1. * output_logprob

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/lm_loss': mask_pad(lm_loss, response_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
            'response/logits': logits,
        }
