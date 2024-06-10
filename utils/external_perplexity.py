# https://huggingface.co/transformers/perplexity.html
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'

import torch
from IPython import embed
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
from tqdm import tqdm

device =  torch.device("cuda:" + str(min(torch.cuda.device_count()-1, 1)) if torch.cuda.is_available() else "cpu")
model_id = 'gpt2-xl'
perp_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
perp_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
perp_model.eval()

def get_external_perplexity(sentences):
    perp = []
    for sentence in tqdm(sentences):
        tokenized = perp_tokenizer(sentence, return_tensors = "pt").to(device)
        with torch.no_grad():
            try:
                output = perp_model(tokenized["input_ids"], labels=tokenized["input_ids"])
            except:
                print("\t\t\t sentence is bad:", sentence)
                continue
        perp.append(math.exp(output.loss.item()))
    return perp

def get_external_probability(sentences):
    perp = []
    for sentence in tqdm(sentences):
        tokenized = perp_tokenizer(sentence, return_tensors = "pt").to(device)
        with torch.no_grad():
            try:
                output = perp_model(tokenized["input_ids"], labels=tokenized["input_ids"])
            except:
                print("\t\t\t sentence is bad:", sentence)
                continue
        perp.append(math.exp(-output.loss.item()))
    return perp

if __name__ == "__main__":    
    print(get_external_perplexity(["Hey, that's not fair", "gihasb", ". .", ".....", "hey hey hey hey", "I'm \\\\\ going \\\\ to \\\\\ the store", "I'm going to the store"]))
    print(get_external_probability(["Hey, that's not fair", "gihasb", ". .", ".....", "hey hey hey hey", "I'm \\\\\ going \\\\ to \\\\\ the store", "I'm going to the store"]))
    embed()
