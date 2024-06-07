import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import collections
from utils.constants import NEGATIVE_INF
from unidecode import unidecode
import re
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')
import re
import torch
T = TypeVar('T')

"""
Functions for training/loss computations
"""
def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)

def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)

def logits_to_entropy(logits):
    try:
        distribution = torch.distributions.Categorical(logits=logits)
    except Exception as e:
        print(e)
        print("Probably getting a CUDA out of memory exception. Check your GPUs and batch size")
        from IPython import embed
        embed()
    return distribution.entropy()

def mask_pad(value, mask):
    return value * mask + NEGATIVE_INF * (1 - mask)

def ceil_div(a, b):
    return (a - 1) // b + 1

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

def batchify_zip(data1, data2, batch_size: int, labels=None, labels2=None):
    assert batch_size > 0
    assert len(data1) == len(data2)

    if labels is None:
        labels = [""] * len(data1)
    if labels2 is None:
        labels2 = [""] * len(data1)

    batch1, batch2, batch3, batch4 = [],[],[],[]
    for item1, item2, item3, item4 in zip(data1, data2, labels, labels2):
        # Yield next batch
        if len(batch1) == batch_size:
            yield batch1, batch2, batch3, batch4
            batch1, batch2, batch3, batch4 = [], [], [], []

        batch1.append(item1)
        batch2.append(item2)
        batch3.append(item3)
        batch4.append(item4)

    # Yield last un-filled batch
    if len(batch1) != 0:
        yield batch1, batch2, batch3, batch4

def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)

def alpha_scheduler(step, base_alpha, interval=2000):
    return base_alpha * pow(1 + step//interval, -0.25)

"""
Functions to clean the outputs and the input data
"""
def clean_output(text):
    # return text.strip("\n").lstrip(".").split("\n")[0].replace("\n"," ").strip()
    return re.sub(r'\s+', ' ', unidecode(text).strip())

def clean_multilabel(text):
    return remove_bad_punc(text.lower())

# Method to specificially clean weird underscores found in joyce
def clean_joyce(text):
    text = text.replace("_", "")
    text = text.strip()
    text = unidecode(text).strip()
    # Shakespeare weird dot dash
    text = text.replace(".--", ". ")
    return text

# Method to specificially clean weird hashtags in coha1990
def clean_coha1990(text):
    text = text.replace("#", "")
    text = text.replace('|', '')
    text = text.strip()
    return text

def clean_coha1890(text):
    text = text.replace('|', '')
    text = text.replace("--(", "-- (")
    text = text.replace(")--", ") --")
    text = text.replace("((", "")
    text = text.replace("( (", "(")
    return text

def clean_coha1810(text):
    text = text.replace("--(", "-- (")
    text = text.replace(")--", ") --")
    return text

def clean_lyrics(text):
    text = re.sub(r'([a-zA-Z])\?([a-z])', r"\1'\2", text)
    text = re.sub(r'([A-Z])\?([A-Z])', r"\1'\2", text)
    text = re.sub(r' \?([a-zA-Z]{1,})\?', r' \1', text)
    return text

def clean_shakespeare(text):
    text = unidecode(text).strip()
    # Shakespeare weird dot dash
    text = text.replace(".--", ". ")
    text = text.replace(".-", ". ")   
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_bad_punc(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    text = unidecode(text).strip()

    text = text.replace("* * * * *", "")
    text = text.replace("%--", "")

    text = text.replace(" .!", "!")
    # regex to get rid of .! when it is after letter
    text = re.sub(r'([a-zA-Z])\.!', r'\1!',text)

    if text.startswith(", \""):
        text = text[3:]
        text = text.strip()

    text = text.replace("// automatically checked by", "")
    # coha 1990 wierd html
    text = text.replace(",?", ",")
    text = text.replace("<p>", "")
    text = text.replace("<P>", "")
    text = text.replace("<br>", "")
    text = text.replace("<BR>", "")
    text = text.replace("&nbsp;", "")
    text = text.replace("&NBSP;", "")
    text = text.replace("// ", " ")
    text = text.replace("- /z/ ", "")
    text = text.replace("/z/", "")
    text = text.replace("/Z/", "")
    text = text.replace("/q/", "")
    text = text.replace("/Q/", "")
    text = text.strip()
    
    # Coha 1810 remove weird colon dash and comma dash and quote dahs
    text = text.replace(";--", "--")
    text = text.replace(":--", "--")
    text = text.replace(",--", ",")
    text = text.replace(":--\"", "--\"")
    text = text.replace("--\"", " \"")
    text = text.replace("\"--", "\" ")

    # Coha 1990 remove weird quotes with space
    text = re.sub(r'(\" ){1,}\"', '\"', text)

    # fix comma quotes issue (coha 1990)
    text = re.sub(r", \"(.*?), \" (.*)", r'," \1, "\2', text)

    # Converts dashes with spaces - - - -
    text = re.sub(r'(- ){1,}-', '--', text)

    # Add space between dash and letters
    text = re.sub(r'--([a-zA-Z])', r'-- \1', text)
    text = re.sub(r'([a-zA-Z])--', r'\1 -', text)

    # Converts repeated symbols to one
    text = re.sub(r'(\-){2,}', '-', text)
    text = re.sub(r',{2,}', ",", text)
    text = re.sub(r':{2,}', ":", text)
    text = re.sub(r';{2,}', ";", text)

    # Romantic poetry (ending with weird symmbols) but also do a check to make sure we're not removing smily faces
    if (text.endswith(":") or text.endswith(";")) and not (text.endswith("(:") or text.endswith("(;")):
        text = text[:-1]
        
    if text.endswith(",") or text.endswith('-'):
        text = text[:-1]
    
    # Remove weird starts
    if (text.startswith(":") or text.startswith(";")) and not (text.startswith(":)") or text.startswith(":-)") or text.startswith(";)") or text.startswith(";-)")):
        text = text[1:]
    
    text = text.strip()

    if text.startswith(",") or text.startswith('-'):
        text = text[1:]
    
    # Convert many dots into one
    text = text.replace(".. ..", "....")
    text = text.replace(".. .", "...")
    text = text.replace(". ..", "...")
    # text = re.sub(r'\.\.\.(\.{1,}$)', '...', text)

    if text.startswith("..") and not text.startswith("..."):
        text = text[2:]
    
    if text.startswith(".") and not text.startswith("..."):
        text = text[1:]

    text = text.strip()

    if text.startswith("**"):
        text = text[2:]
    if text.startswith("*"):
        text = text[1:]

    text = text.strip()

    if text.startswith(")"):
        text = text[1:]

    text = text.strip()

    # ENGLISH TWEET FIXING
    # Remove blank hashtags
    text = re.sub(r'(# ){1,}#', '#', text)
    # Remove weird hashtag with underscores
    text = re.sub(r'#(_{2,})', "", text)
    # Remove multiple hashtags
    text = re.sub(r'#{2,}', "#", text)
    # Remove ending with hashtag
    if text.endswith("#"):
        text = text[:-1]
    text = text.strip()

    # Fix quation marks issue (Aae, coha1990)
    if text.startswith("\"") and text.count("\"") == 1:
        text = text[1:]
    text = text.strip()
    if text.endswith("\"") and text.count("\"") == 1:
        text = text[:-1]
    text = text.strip()
    if text.startswith("'") and text.count("'") == 1:
        text = text[1:]
    text = text.strip()
    if text.endswith("'") and text.count("'") == 1:
        text = text[:-1]
    text = text.strip()

    if text.endswith(").") and text.count("(") == 0:
        text = text[:-2]
        text = text + "."

    #aae get rid of ending backslahes (also the \\ is treated as one character)
    if text.endswith("\\"):
        text = text[:-1]

    # switchbaord get rid of weird _1
    text = text.replace("_1", "")
    text = text.replace("<b_aside>", "")
    text = text.replace("<e_aside>", "")

    # aae fix weird dialogue with extra slash
    text = text.replace("\\\"", "\"")

    # Finally cleaning (probably redundant)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    #coha1810
    text = text.replace(" ,-", ",")
    text = text.replace(", -", ",")
    text = re.sub(r',([a-zA-Z])', r', \1', text)

    # Joyce weird parenths, also preserve aae emoticons
    if text.endswith("(") and not text.endswith(":(") and not text.endswith(";(") and not text.endswith("-("):
        text = text[:-1]
    if text.endswith(")") and text.count("(") == 0 and not text.endswith(";)") and not text.endswith(":)") and not text.endswith("-)"):
        text = text[:-1]
    if text.startswith("(") and text.count(")") == 0 and not text.startswith("(-") and not text.startswith("(:") and not text.startswith("(;"):
        text = text[1:]

    # Moses detoknizer to hopefully fix weird spacing issues
    text = clean_tokenization(text.strip())

    # Space is important, looking for extra or non-matching quotes
    if text.endswith(" \""):
        text = text[:-2]
    if text.endswith(" '"):
        text = text[:-2]
        
    #coha1810
    text = text.replace(";-", ";")
    text = text.replace("; -", ";")

    # Fix more spacing
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")

    # Romantic poetry (ending with weird symmbols) but also do a check to make sure we're not removing smily faces
    if (text.endswith(":") or text.endswith(";")) and not (text.endswith("(:") or text.endswith("(;")):
        text = text[:-1]

    return text.strip()

def clean_tokenization(text):
    return md.detokenize(text.split())

def convert_label_list(list_of_dics, cur_key = 'label', cur_val = 'score', label_map = None):
    output = {}
    for item in list_of_dics:
        output[label_map[int(item[cur_key].split("_")[1])]] = item[cur_val]
    return output

def output_folder_name(args):
    ae_str = "ae" if args.use_antiexpert else "noae"
    random_data_str = "nonrandom" if args.nonrandom_split else "random" + str(args.seed)
    return "_".join([
        args.target_style, str(args.num_examples), str(args.dataset_partition),
        str(args.alpha),  str(args.filter_p), ae_str, 
        str(args.temperature), str(args.no_repeat_ngrams),
        random_data_str
        ])


def expand_and_prepend_tokens(input_ids, attention_mask, tokens_to_prepend_list, fill_value = 50256):
    """
    Expand input_ids and attention_mask tensors, and prepend varying tokens to the start of the actual sequences.
    
    Parameters:
    - input_ids (torch.Tensor): Tensor of input IDs with left padding.
    - attention_mask (torch.Tensor): Tensor indicating which tokens are padding (0) and which are not (1).
    - tokens_to_prepend_list (list of lists): Each sublist contains tokens to prepend to each corresponding instance.
    
    Returns:
    - torch.Tensor: Expanded and updated input_ids with varying tokens prepended.
    - torch.Tensor: Expanded and updated attention mask.
    """
    n = len(tokens_to_prepend_list[0])  # Assuming all sublists are the same length
    new_size = input_ids.size(1) + n
    
    new_input_ids = torch.full((input_ids.size(0), new_size), fill_value=fill_value, dtype=input_ids.dtype, device=input_ids.device)
    new_attention_mask = torch.zeros((attention_mask.size(0), new_size), dtype=attention_mask.dtype, device=attention_mask.device)

    for i in range(input_ids.size(0)):
        tokens_to_prepend = torch.tensor(tokens_to_prepend_list[i], device=input_ids.device, dtype=input_ids.dtype)
        
        seq_start = torch.where(attention_mask[i] == 1)[0][0]
        insert_pos = seq_start + n
        
        # Insert the tokens to prepend for this instance
        new_input_ids[i, seq_start:insert_pos] = tokens_to_prepend
        new_attention_mask[i, seq_start:insert_pos] = 1
        
        # Copy the original sequence into the new tensors
        new_input_ids[i, insert_pos:insert_pos+input_ids.size(1)-seq_start] = input_ids[i, seq_start:]
        new_attention_mask[i, insert_pos:insert_pos+attention_mask.size(1)-seq_start] = attention_mask[i, seq_start:]
        
    return new_input_ids, new_attention_mask