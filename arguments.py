import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument("--save_naming", type=str, default = None)

    # dataset
    parser.add_argument(
        '--target-sentiment', type=str, default='positive')
    parser.add_argument(
        '--output_dir', type=str, default='outputs')
    parser.add_argument(
        '--dataset_dir', type=str, default='datasets/cds',
        help='JSONL file containing train prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--precomputed_dataset_dir', type=str, default=None,
        help='dataset if we precomputed generations')
    parser.add_argument("--num_examples_val", type = int, default = 1000)
    parser.add_argument("--num_examples_train", type = int, default = 1000)
    parser.add_argument(
        '--nonrandom_split', action='store_true')
    parser.add_argument(
        '--dataset_partition', default=0, type=int, help='overriding paremeter to specify specific dataset partition')

    # reward
    parser.add_argument(
        '--n_extra_tokens', type=int, default=5, help='number of reward categorization')
    parser.add_argument(
        '--sample_interval', type=int, default=2000, help='step interval to sample from current policy') #500
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    parser.add_argument(
        '--reward_model_dir', type=str, default='models/multilabel/03-21-2023_21:54:15/checkpoint-4272')

    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.05, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.06, help='coefficient for entropy term in reward')
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')

    # policy / style experts
    parser.add_argument(
        '--init_model', type=str, default='models/paraphraser_gpt2_large', help='language model used for policy.') # gpt2-large
    parser.add_argument(
        '--ref_model', type=str, default='models/paraphraser_gpt2_large', help='language model used for reference policy.') # gpt2-large
    parser.add_argument(
        '--expert_dir', type=str, default='models/style_gpt2_experts_gpt2large_newpreproc/gpt2_large_preprocessed_eos', help='path to expert models.')
    parser.add_argument(
        '--use_experts', action='store_true', help='whether to use expert style lms')
    parser.add_argument(
        '--use_antiexpert', action='store_true')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')

    # training
    parser.add_argument(
        '--total_episodes', type=int, default=40000, help='total number of episodes') #3000000
    parser.add_argument(
        '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=500, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max_grad_norm', type=float, default=0.5, help='maximum norm of gradients ')
    parser.add_argument(
        '--max_gen_length', type = int, default = 30, help = 'max generation length')

    # generation

    # TODO: outdated but we COULD use this
    parser.add_argument(
        '--num_samples', type=int, default=25, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top_p', type=float, default=1.0, help='hyperparameter for nucleus sampling in final decoding')
    parser.add_argument(
        '--filter_p', type=float, default=0.6, help='hyperparameter for nucleus sampling in base lm')
    parser.add_argument(
        '--alpha', type=float, default=0.4, help='hyperparameter for mixing base and expert lm probs')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
        '--save_interval', type=int, default=500, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda_deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--no_repeat_ngrams', type = int, default = 3, help = 'no_repeat_ngrams amount')
    parser.add_argument(
        '--sample', action='store_true')
    parser.add_argument(
        '--multiple_reward_tokens', action='store_true')

    # Number of checkpoints to save
    parser.add_argument(
        '--save_best', type=int, default=5)

    parser.add_argument(
        '--load_from_ckpt', type=str, default=None)

    parser.add_argument(
        '--sim_weight', type=float, default=1.0)
    parser.add_argument(
        '--style_weight', type=float, default=1.0)
    parser.add_argument(
        '--flu_weight', type=float, default=1.0)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
