# STEERStyleTransfer

This is the repository for the 2023 EMNLP Findings Paper, ["🐂 STEER: Unified Style Transfer with Expert Reinforcement"](https://arxiv.org/abs/2311.07167)

<p align="center">
  <img src="steer_method.png" alt="drawing" width="100%"/>
</p>


# Getting Started

## Setting up the Environment
To set up the environment to run the code, make sure to have conda installed, then run

```
conda env create -f environment.yml
```
Then, activate the environment:

```
conda activate style
```

## Downloading required resources

We need to download a few resources for STEER.

*  **Base Model**

    Download the GPT2-large paraphraser from [Google Drive](https://drive.google.com/drive/folders/1hB0lJt4MjuWbgdY7_I_2eISAoNmM6O9f); it is the folder named `paraphraser_gpt2_large`. Place this model in the following directory: `models/paraphraser_gpt2_large`. (TODO: upload this to huggingface)

* **CDS Synthetic Parallel Training Dataset** 

    Our filtered dataset is located on [HuggingFace](https://huggingface.co/datasets/hallisky/STEER-data-top-400k-combo/tree/main). However, since the huggingface `load_dataset` function has an issue, we must manually download `top400k_data_combo.jsonl`. Place the data in the following directory: `training_data_topk/top400k_data_combo.jsonl`.
  
### (Optional) Downloading resources for offline use

Though the STEER code will automatically download models as the code progresses, we can also pre-cache these values by downloading them first. Use the following code to download the other models used in the code:

```
python3 download_all_models.py
```

If you do this, make sure the `cache_dir` in `main.py` matches the `cache_dir` specified in `download_all_models.py`.

### (Optional) Creating own dataset

You can also use  your own data from authors with STEER.

First, note the reward model for target style is based on the CDS style data from the STEER data. You most likely want to re-train the dataset using `style_classifier/train_multilabel.py`. Change lines 40-77 to load and preprocess your own data. See `style_classifier/README.md` for more details on training.

After this, there are two options.

1. Dataset already has parallel pairs of the same text from different authors

* Note * this part requires use of `vllm`, a fast inference library, so you will need to run `pip install vllm` in your conda environment.

Although unlikely, if you already have parallel data from different authors to a target author, you can directly convert your data to the format required for STEER.

We have an example script in `create_data/create_from_parallel.py` for this case with example data (`test_parallel.json`). Essentially, we need to load in the data, score it by running 1) similarity scores 2) fluency scores and 3) target style scores. We then compute an overall score for each data point by multiplying these metrics together. Then, we filter out the top-k data instances by score and save it with the correct columns. 

You can run this code and substitute your input data path, the output save path, the top-k to filter, and your new classifier model. You will also need to make sure the `target_styles`` input into `get_scores` match the target styles in the newly trained model.

2. We have different texts from each author

In this case, we will need to generate synthetic, parallel data from each author in the corpus to get data to the format required for STEER.

The part that differs from the data process in 1) is that we need to *generate* synthetic parallel rewrites from one author to another. A way we can do this is to use few-shot prompting with a large model like Llama-3-8b.

We have an example script at `create_data/create_from_nonparallel.py` for this case with example data (`test_noonparallel.json`). We essentially iterate through texts from each author, and use the language model to make candidate rewrites given a few-shot prompt in another author's style.

Then, as in 1) above, we score, filter, and save.

## Compute Requirements

## Training

To train our unified style model, we can run the following command:

```
# For a100, 200k
python3 main.py \
    --precomputed_dataset_dir hallisky/STEER-data-top-400k-combo \
    --n_extra_tokens 5 \
    --init_model models/paraphraser_gpt2_large \
    --ref_model models/paraphraser_gpt2_large \
    --temperature 1.0 \
    --reward_model_dir hallisky/cds_style_classifier \
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
```

# Citing this work
If you use/reference this work, please cite us with:

```
@inproceedings{hallinan-etal-2023-steer,
    title = "{STEER}: Unified Style Transfer with Expert Reinforcement",
    author = "Hallinan, Skyler  and
      Brahman, Faeze  and
      Lu, Ximing  and
      Jung, Jaehun  and
      Welleck, Sean  and
      Choi, Yejin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.506",
    doi = "10.18653/v1/2023.findings-emnlp.506",
    pages = "7546--7562",
    abstract = "While text style transfer has many applications across natural language processing, the core premise of transferring from a single source style is unrealistic in a real-world setting. In this work, we focus on arbitrary style transfer: rewriting a text from an arbitrary, unknown style to a target style. We propose STEER: Unified Style Transfer with Expert Reinforcement, a unified frame-work developed to overcome the challenge of limited parallel data for style transfer. STEER involves automatically generating a corpus of style-transfer pairs using a product of experts during decoding. The generated offline data is then used to pre-train an initial policy before switching to online, off-policy reinforcement learning for further improvements via fine-grained reward signals. STEER is unified and can transfer to multiple target styles from an arbitrary, unknown source style, making it particularly flexible and efficient. Experimental results on a challenging dataset with text from a diverse set of styles demonstrate state-of-the-art results compared to competitive baselines. Remarkably, STEER outperforms the 175B parameter instruction-tuned GPT-3 on overall style transfer quality, despite being 226 times smaller in size. We also show STEER is robust, maintaining its style transfer capabilities on out-of-domain data, and surpassing nearly all baselines across various styles. The success of our method highlights the potential of RL algorithms when augmented with controllable decoding to overcome the challenge of limited data supervision.",
}
```
