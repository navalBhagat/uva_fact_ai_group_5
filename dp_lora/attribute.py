import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

def get_attr_dict(model, data_loader, device):
    model = model.to(device)
    model.eval()

    activations = {}  # layer_name -> list[tensor]
    order_of_modules = []

    def hook_fn(name):
        def _hook(module, input, output):
            if name not in activations:
                activations[name] = [[], type(module)]
            if isinstance(module, VectorQuantizer):
                _, _, (_, _, min_encoding_indices) = output
                min_encodings = F.one_hot(min_encoding_indices, num_classes=module.n_e).float()
                out = min_encodings.detach().cpu()
            else:
                out = output.detach().cpu()
            activations[name][0].append(out)
            order_of_modules.append((name, module))
        return _hook

    hooks = []
    skip_modules = set()
    for name, layer in model.named_modules():
        if isinstance(layer, VectorQuantizer):
            hooks.append(layer.register_forward_hook(hook_fn(name)))
            skip_modules.update(layer.modules())
        if layer not in skip_modules and (len(list(layer.children())) == 0 and any(p.requires_grad for p in layer.parameters())):
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for img in data_loader:
            img = img.to(device)
            out = model(img)

    for name in activations:
        activations[name][0] = torch.cat(activations[name][0], dim=0)

    filter_values = {}  # layer_name -> np array [C]

    for name, [activation, dt] in activations.items():
        # [N, C, H, W] → [C]
        if dt == VectorQuantizer:
            filter_values[name] = activation.mean(dim=0).numpy()
        else:
            filter_values[name] = activation.mean(dim=(0, 2, 3)).numpy()

    kept_filters = {}  # layer_name -> bool[C]

    for name, module in model.named_modules():
        if name not in filter_values:
            continue
            
        scores = np.abs(filter_values[name])
        threshold = np.quantile(scores, 0.75)
        kept_filters[name] = scores > threshold
            
    param_2_mask = {}
    prev_keep = None
    skip_keeps = []

    for name, module in order_of_modules:
        if name not in kept_filters:
            continue
        
        weight = module.embedding.weight if isinstance(module, VectorQuantizer) else module.weight
        curr_keep = kept_filters[name].astype(np.float32)
        bias_boolean = None
        
        if isinstance(module, VectorQuantizer):
            # VectorQuantizer layer
            in_c, out_c = weight.shape
            input_keep = np.ones(out_c, dtype=np.float32)
            mask = input_keep[None, :] * curr_keep[:, None]
            mask = np.broadcast_to(mask, (in_c, out_c))
            prev_keep = None

        elif isinstance(module, nn.GroupNorm):
            num_c = weight.shape[0]
            input_keep = np.ones(num_c, dtype=np.float32) if prev_keep is None else prev_keep
            mask = curr_keep[:,None] * input_keep[:,None]
            mask = np.broadcast_to(mask, (num_c))
            bias_boolean = mask
            prev_keep = curr_keep
            
        else:
            out_c, in_c, k1, k2 = weight.shape
            input_keep = np.ones(in_c, dtype=np.float32) if prev_keep is None else prev_keep
            mask = curr_keep[:, None, None, None] * input_keep[None, :, None, None]
            mask = np.broadcast_to(mask, (out_c, in_c, k1, k2))
            prev_keep = curr_keep
            bias_boolean = curr_keep

        param_2_mask[name + ".weight"] = torch.tensor(mask, device=weight.device, dtype=weight.dtype)
        param_2_mask[name + ".bias"] = bias_boolean
    
    return param_2_mask