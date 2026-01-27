import argparse
import random

import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
import torch
from torchvision.utils import save_image
from codecarbon import OfflineEmissionsTracker
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.rank_scheduler import get_rank_scheduler


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    try:
        config.model.params.ignore_keys = []
        config.model.params.ckpt_path = None
    except ConfigAttributeError:
        pass
    model = instantiate_from_config(config.model)
    
    # Try to apply rank scheduler if configured (for adaptive LoRA ranks)
    try:
        rank_scheduler_config = config.lightning.callbacks.rank_scheduler.params.scheduler_config
        if rank_scheduler_config is not None:
            print(f"[INFO]: Applying rank scheduler from config: {dict(rank_scheduler_config)}")
            
            # Create the rank scheduler with the same config used during training
            scheduler = get_rank_scheduler(
                rank_scheduler_config.name,
                initial_rank=rank_scheduler_config.initial_rank,
                final_rank=rank_scheduler_config.final_rank,
                schedule_type=rank_scheduler_config.schedule_type
            )
            
            # Find all LoRA modules in the model
            lora_modules = []
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and isinstance(module.lora_A, dict):
                    lora_modules.append((name, module))
            
            if lora_modules:
                num_layers = len(lora_modules)
                scheduler.set_num_layers(num_layers)
                print(f"[INFO]: Found {num_layers} LoRA modules, resizing to match scheduler ranks")
                
                # Apply rank-based resizing to each LoRA module
                for idx, (name, module) in enumerate(lora_modules):
                    target_rank = scheduler.get_rank_for_layer(idx)
                    
                    # Resize lora_A and lora_B tensors for each adapter
                    for adapter_name, lora_a in module.lora_A.items():
                        current_rank = lora_a.weight.shape[0]
                        
                        if target_rank < current_rank:
                            # Truncate to target rank
                            new_a = lora_a.weight.data[:target_rank, :].clone()
                            lora_a.weight.data = new_a
                            lora_a.out_features = target_rank
                            
                            # Also resize lora_B
                            lora_b = module.lora_B[adapter_name]
                            new_b = lora_b.weight.data[:, :target_rank].clone()
                            lora_b.weight.data = new_b
                            lora_b.in_features = target_rank
                            
                            if idx < 3 or idx >= num_layers - 2:
                                print(f"  Layer {idx}: resized rank {current_rank} -> {target_rank}")
    except (ConfigAttributeError, KeyError, AttributeError, TypeError) as e:
        print(f"[INFO]: No valid rank scheduler config found ({type(e).__name__}), using default initialization")
    
    # Load state dict with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    
    if missing_keys:
        lora_missing = [k for k in missing_keys if 'lora' in k.lower()]
        if lora_missing:
            print(f"[WARNING]: {len(lora_missing)} LoRA keys not loaded (expected if ranks differ)")
    
    model.cuda()
    model.eval()
    return model


def get_uc(model, uc_class):
    x = {model.cond_stage_key: torch.tensor([uc_class], device=model.device)}
    return model.get_learned_conditioning(x)


def get_c(model, args):
    if args.prompt is not None:
        return model.get_learned_conditioning([args.prompt])
    if args.classes is not None:
        all_classes = torch.tensor(args.classes, device=model.device)
        all_conditioning = []
        for batch_classes in all_classes.split(args.batch_size):
            c = model.get_learned_conditioning({model.cond_stage_key: batch_classes})
            all_conditioning.append(c)
        return torch.vstack(all_conditioning)
    else:
        return None


def main(args):
    print("[INFO]: Set all seeds to", args.seed)
    set_seeds(args.seed)

    config = OmegaConf.load(args.yaml)
    model = load_model_from_config(config, args.ckpt)

    num_samples_per_class = args.num_samples // len(args.classes)
    print(f"[INFO]: Will generate {num_samples_per_class} samples per class for {len(args.classes)} classes")
    
    # code carbon
    # Create tracker early so it knows where to write emissions.csv (inside logdir)
    carbon_tracker = OfflineEmissionsTracker(
        country_iso_code="NLD",   # Snellius is in Netherlands
        output_dir="output/emissions/conditional_sampling",        # emissions.csv goes into this folder
        save_to_file=True,
        log_level="info",
    )
    carbon_tracker.start()

    with model.ema_scope():
        # Get the conditioning information for each label
        conditioning = get_c(model, args)
        print("[INFO]: Got conditioning information of shape", conditioning.shape)

        # Get unconditional conditioning
        uc = get_uc(model, args.uc) if args.uc is not None else None

        # Set up sampler
        sampler = DDIMSampler(model)
        shape = [model.model.diffusion_model.in_channels,
                 model.model.diffusion_model.image_size,
                 model.model.diffusion_model.image_size]
        all_samples = []

        # Sample images
        class_indices = torch.tensor(range(len(args.classes))).repeat_interleave(num_samples_per_class)
        for batch_classes in class_indices.split(args.batch_size):
            batch_size = batch_classes.shape[0]
            batch_conditioning = conditioning[batch_classes]
            batch_uc = uc.repeat_interleave(batch_size, dim=0) if args.uc is not None else None
            samples, _ = sampler.sample(S=args.ddim_steps,
                                        batch_size=batch_size,
                                        shape=shape,
                                        conditioning=batch_conditioning,
                                        unconditional_conditioning=batch_uc,
                                        unconditional_guidance_scale=args.uc_scale,
                                        verbose=False,
                                        eta=args.eta)
            for batch_samples in samples.split(args.decoder_batch_size or batch_size):
                batch_samples = model.decode_first_stage(batch_samples).cpu()
                all_samples.append(batch_samples)

    emissions = carbon_tracker.stop()
    print("CodeCarbon emissions (kg CO2):", emissions)
    
    # Collate images and save
    images = torch.vstack(all_samples)
    if args.output.endswith(".pt"):
        labels = torch.tensor(args.classes).repeat_interleave(num_samples_per_class)
        result = {'image': images, 'class_label': labels}
        torch.save(result, args.output)
    elif args.output.endswith(".png"):
        save_image(images * 0.5 + 0.5, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--yaml", type=str, required=True, help="Path to the model YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to output file (.pt|.png)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=20, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of samples to generate per batch")
    parser.add_argument("--decoder_batch_size", type=int, default=None, help="Number of samples to decode per batch")
    grp_cond = parser.add_argument_group("Conditional Generation")
    grp_cond.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to generate samples for")
    grp_cond.add_argument("--prompt", type=str, nargs="?", default=None, help="A prompt to generate samples for")
    grp_cond.add_argument("--uc", type=int, default=None, help="Class to use for unconditional guidance")
    grp_cond.add_argument("--uc_scale", type=float, default=None, help="Unconditional guidance scale")
    grp_ddim = parser.add_argument_group("DDIM Options")
    grp_ddim.add_argument("--ddim_steps", type=int, default=200, help="number of steps for ddim sampling")
    grp_ddim.add_argument("--eta", type=float, default=1.0, help="eta for ddim sampling")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
