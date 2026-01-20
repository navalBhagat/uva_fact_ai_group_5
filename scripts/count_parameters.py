"""
Script to count total and trainable parameters in a model checkpoint,
specifically focusing on LoRA parameters.
"""

import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location='cpu')

    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'model' in state_dict:
        state_dict = state_dict['model']

    total = 0
    trainable = 0

    for name, p in state_dict.items():
        if not isinstance(p, torch.Tensor):
            continue

        num_params = p.numel()
        total += num_params

        name_low = name.lower()
        if 'lora_' in name_low and '.weight' in name_low:
            if not any(extra in name_low for extra in ['optimizer', 'exp_avg',
                                                       'step']):
                trainable += num_params

    print("--- Parameter Count Report ---")
    print(f"Total parameters:      {total:,}")
    print(f"Trainable (LoRA):      {trainable:,}")
    print(f"Percentage Trainable:  {(100 * trainable / total):.4f}%")
    print("------------------------------")
