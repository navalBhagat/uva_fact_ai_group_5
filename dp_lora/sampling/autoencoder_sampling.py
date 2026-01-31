import argparse
from math import ceil
import random

import numpy as np
from omegaconf import OmegaConf
import torch
import torchvision

from ldm.util import instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def test_reconstruction(model, img_tensor):
    
    with torch.no_grad():
        # Encode
        posterior = model.encode(img_tensor)
        z = posterior.sample()
        print(f"Encoded latent shape: {z.shape}")
        print(f"Latent stats - mean: {z.mean():.3f}, std: {z.std():.3f}")
        
        # Decode
        reconstruction = model.decode(z)
        
        # Save
        original = torch.clamp((img_tensor[0] + 1.0) / 2.0, 0.0, 1.0)
        recon = torch.clamp((reconstruction[0] + 1.0) / 2.0, 0.0, 1.0)
        
        torchvision.utils.save_image(original, "original.png")
        torchvision.utils.save_image(recon, "reconstruction.png")
        
        print("Saved original.png and reconstruction.png")

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default=None, help="Path to the config file for the model")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file (.pt)")
    parser.add_argument("--seed", type=int, default=None, help="Number of DDIM steps")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of samples to generate per batch")
    args = parser.parse_args()

    if args.seed is not None:
        set_seeds(args.seed)

    config = OmegaConf.load(args.yaml)
    model = load_model_from_config(config, args.ckpt).to('cuda')
    # parameters = torch.randn((args.batch_size, 6, 64, 64))
    # sampler = DiagonalGaussianDistribution(parameters)
    # data = instantiate_from_config(config.data)
    # data.prepare_data()
    # data.setup()
    # img = data.datasets['train'][0]["image"]
    # test_reconstruction(model, img.permute(2,0,1).unsqueeze(0).to('cuda'))
    iters = ceil(args.num_samples / args.batch_size)
    samples_list = list()
    
    with torch.no_grad():
        for i in range(iters):
            sample = torch.randn((args.batch_size, 3, 64, 64))
            output = model.decode(sample.to(model.device))
            samples_list.append(output.cpu())
    
    samples = torch.cat(samples_list, dim=0)
    dic = {"image": samples}
    torch.save(dic, args.output)

if __name__ == "__main__":
    # Add to your main():
    main()
    