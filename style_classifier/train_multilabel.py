import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
os.environ["WANDB_DISABLED"] = "true" 

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from utils.utils import clean_output, clean_multilabel, clean_joyce, clean_coha1990, clean_coha1890, clean_coha1810, clean_lyrics, clean_shakespeare

from IPython import embed
import random
import math
from datetime import datetime
time = datetime.now()    
date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
import json

def main(args):
    if not args.evaluate: # Train model from scratch
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large", num_labels=args.num_labels, problem_type="multi_label_classification"
        )
    else: # Load existing model for evaluation only
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_path, num_labels=args.num_labels, problem_type="multi_label_classification"
        )
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    train_data, train_labels = [], []
    dev_data, dev_labels = [], []
    style_dict, rev_style_dict = {}, {}

    # Load in the train data
    for i, style in enumerate(os.listdir(args.data_dir)):
        style_dict[style] = i
        rev_style_dict[i] = style

        with open(os.path.join(args.data_dir, style, f"tmp/train.txt"), 'r') as f:
            train_cur = [clean_output(s.strip()) for s in f.readlines()]
        
        with open(os.path.join(args.data_dir, style, f"tmp/dev.txt"), 'r') as f:
            dev_cur = [clean_output(s.strip()) for s in f.readlines()]

        random.seed(0)
        train_cur = random.sample(train_cur, args.train_len)
        dev_cur = random.sample(dev_cur, args.dev_len)

        # Cleaning the data
        if style == "joyce":
            train_cur = [clean_joyce(t) for t in train_cur]
            dev_cur = [clean_joyce(t) for t in dev_cur]
        elif style == "coha_1990":
            train_cur = [clean_coha1990(t) for t in train_cur]
            dev_cur = [clean_coha1990(t) for t in dev_cur]
        elif style == "coha_1890":
            train_cur = [clean_coha1890(t) for t in train_cur]
            dev_cur = [clean_coha1890(t) for t in dev_cur]
        elif style == "coha_1810":
            train_cur = [clean_coha1810(t) for t in train_cur]
            dev_cur = [clean_coha1810(t) for t in dev_cur]
        elif style == "lyrics":
            train_cur = [clean_lyrics(t) for t in train_cur]
            dev_cur = [clean_lyrics(t) for t in dev_cur]
        elif style == "shakespeare":
            train_cur = [clean_shakespeare(t) for t in train_cur]
            dev_cur = [clean_shakespeare(t) for t in dev_cur]
        # Additional cleaning for multilabel task
        train_cur = [clean_multilabel(t) for t in train_cur]
        dev_cur = [clean_multilabel(t) for t in dev_cur]

        train_data.extend(train_cur)
        train_labels.extend([i] * len(train_cur))

        dev_data.extend(dev_cur)
        dev_labels.extend([i] * len(dev_cur))

    # Save dictionary with label ids to style
    with open(os.path.join(args.output_dir, "styledict.json"), "w") as k:
        k.write(json.dumps(style_dict))

    with open(os.path.join(args.output_dir, "revstyledict.json"), "w") as m:
        m.write(json.dumps(rev_style_dict))

    train_labels = torch.nn.functional.one_hot(torch.tensor(train_labels), num_classes=args.num_labels).type(torch.float32)
    dev_labels = torch.nn.functional.one_hot(torch.tensor(dev_labels), num_classes=args.num_labels).type(torch.float32)

    # Shuffle the data
    train_combo = list(zip(train_data, train_labels))
    random.shuffle(train_combo)
    train_data, train_labels = zip(*train_combo)
    train_data, train_labels = list(train_data), list(train_labels)

    dev_combo = list(zip(dev_data, dev_labels))
    random.shuffle(dev_combo)
    dev_data, dev_labels = zip(*dev_combo)
    dev_data, dev_labels = list(dev_data), list(dev_labels)
    
    # Collate function for batching tokenized texts
    def collate_tokenize(data):
        text_batch = [element["text"] for element in data]
        tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
        label_batch = [element["label"] for element in data]
        tokenized['labels'] = torch.stack(label_batch)
        
        return tokenized

    class StyleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __getitem__(self, idx):
            item = {}
            item['text'] = self.texts[idx]
            item['label'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = StyleDataset(train_data, train_labels)
    dev_dataset = StyleDataset(dev_data, dev_labels)

    # Number of steps per epoch
    steps = len(train_labels)/(args.batch_size*torch.cuda.device_count())
    # Save every quarter epoch
    save_steps = math.ceil(steps / args.save_ratio)
    print(save_steps)

    # Training branch
    if not args.evaluate:
        compute_metrics= None
        metric_for_best_model = None
        greater_is_better = None

        # If we want to calculate classification accuracy while we're training
        if args.use_accuracy_for_training:
            def accuracy(eval_pred):
                predictions, labels = eval_pred
                predictions = torch.argmax(torch.tensor(predictions), dim=-1).tolist()
                labels = [a[1].item() for a in torch.nonzero(torch.tensor(labels))]
                return {'acc': sum([a == b for a, b in zip(predictions, labels)])/len(predictions)}

            compute_metrics = accuracy
            metric_for_best_model = 'acc'
            greater_is_better = True

        args = TrainingArguments(
            output_dir = os.path.join(args.output_dir,date_time), 
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy='steps',
            num_train_epochs=args.epochs,
            eval_steps = save_steps,
            save_steps = save_steps,
            logging_steps = save_steps,
            lr_scheduler_type = 'linear',
            learning_rate=args.lr,
            seed = args.seed,
            warmup_ratio = 0.1,
            load_best_model_at_end = True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            remove_unused_columns=False
            )

        trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=dev_dataset, 
            tokenizer=tokenizer,
            data_collator = collate_tokenize,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(5)]
            )

        trainer.train()
    else: # Evaluation only branch
        from torch.utils.data import DataLoader
        dataload = DataLoader(dev_dataset, collate_fn = collate_tokenize, batch_size = args.batch_size)
        truth, pred = [], []
        for d in dataload:
            true_labs = [a[1].item() for a in d["labels"].nonzero()]
            truth.extend(true_labs)
            pred.extend(torch.argmax(model(**d).logits, dim=-1).tolist())

        print(sum([a == b for a, b in zip(truth, pred)])/len(pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_ratio', type=int, default=4)
    parser.add_argument(
        '--lr', type=float, default=5e-5)
    parser.add_argument(
        '--epochs', type=int, default=5)
    parser.add_argument(
        '--batch_size', type=int, default=64)
    parser.add_argument(
        '--train_len', type=int, default=24852)
    parser.add_argument(
        '--dev_len', type=int, default=1313)
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--num_labels', type=int, default=11)
    parser.add_argument(
        '--data_dir', type=str, default='datasets/cds', help='datadir')
    parser.add_argument(
        '--output_dir', type=str, default='models/multilabel')
    parser.add_argument(
        '--pretrained_path', type=str, default=None)
    parser.add_argument(
        "--evaluate", action="store_true")

    # If you want eval metric to be accuracy 
    parser.add_argument(
        "--use_accuracy_for_training", action="store_true")
    main(parser.parse_args())
