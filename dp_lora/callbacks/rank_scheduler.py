"""
PyTorch Lightning callback for layer-wise LoRA rank assignment.
Assigns fixed ranks to layers based on their depth in the model.
"""

import logging
from typing import Optional, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from ldm.rank_scheduler import get_rank_scheduler, LayerWiseRankScheduler

logger = logging.getLogger(__name__)


class RankSchedulerCallback(Callback):
    """
    Callback to assign fixed LoRA ranks based on layer position.
    Does NOT schedule ranks over time - each layer gets a fixed rank.
    """
    
    def __init__(
        self,
        scheduler_config: Dict,
        adapter_name: str = "default",
        target_layers: Optional[list] = None,
        log_interval: int = 100,
        **kwargs
    ):
        """
        Args:
            scheduler_config: Dict with keys: name, initial_rank, final_rank, schedule_type
                Example: {"name": "layer-wise", "initial_rank": 16, "final_rank": 4, "schedule_type": "linear"}
            adapter_name: Name of the LoRA adapter
            target_layers: List of layer names to update. If None, updates all LoRA layers.
            log_interval: How often to log
        """
        super().__init__()
        
        self.adapter_name = adapter_name
        self.target_layers = target_layers
        self.log_interval = log_interval
        
        # Create scheduler from config
        config = dict(scheduler_config)
        name = config.pop("name", "layer-wise")
        self.scheduler = get_rank_scheduler(name, **config)
        
        if not isinstance(self.scheduler, LayerWiseRankScheduler):
            raise ValueError(f"Only 'layer-wise' scheduler is supported, got {name}")
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Initialize layer-wise ranks at the start of training."""
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module
        num_lora_layers = self._count_lora_layers(model)
        self.scheduler.set_num_layers(num_lora_layers)
        
        # Apply ranks to all LoRA layers
        self._apply_layer_wise_ranks(model)
        
        # Log configuration
        print("[RankScheduler] Layer-wise rank assignment initialized")
        print(f"[RankScheduler] Found {num_lora_layers} LoRA layers")
        print(f"[RankScheduler] Schedule type: {self.scheduler.schedule_type}")
        print(f"[RankScheduler] Initial rank: {self.scheduler.initial_rank}, Final rank: {self.scheduler.final_rank}")
        
        # Print sample ranks
        if num_lora_layers > 0:
            ranks = [
                (0, self.scheduler.get_rank_for_layer(0)),
                (num_lora_layers // 2, self.scheduler.get_rank_for_layer(num_lora_layers // 2)),
                (num_lora_layers - 1, self.scheduler.get_rank_for_layer(num_lora_layers - 1)),
            ]
            print("[RankScheduler] Sample layer ranks:")
            for layer_idx, rank in ranks:
                print(f"  Layer {layer_idx}: rank {rank}")
    
    def _apply_layer_wise_ranks(self, model) -> None:
        """Apply layer-wise ranks to all LoRA layers in the model."""
        layer_index = 0
        
        for name, module in model.named_modules():
            # Skip if not a layer that might have rank info
            if not (hasattr(module, 'r') or hasattr(module, 'lora_A')):
                continue
            
            # Filter by target layers if specified
            if self.target_layers is not None:
                if not any(layer_pattern in name for layer_pattern in self.target_layers):
                    continue
            
            # Get the fixed rank for this layer
            rank = self.scheduler.get_rank_for_layer(layer_index)
            
            # Update rank dict if it exists
            if hasattr(module, 'r') and isinstance(module.r, dict):
                if self.adapter_name in module.r:
                    module.r[self.adapter_name] = rank
            
            # Update scaling based on alpha and rank
            if hasattr(module, 'alpha') and hasattr(module, 'scaling'):
                if isinstance(module.alpha, dict) and self.adapter_name in module.alpha:
                    alpha = module.alpha[self.adapter_name]
                    if isinstance(module.scaling, dict):
                        module.scaling[self.adapter_name] = alpha / rank
            
            layer_index += 1
    
    def _count_lora_layers(self, model) -> int:
        """Count the number of LoRA layers in the model."""
        count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'r') or hasattr(module, 'lora_A'):
                # Filter by target layers if specified
                if self.target_layers is not None:
                    if any(layer_pattern in name for layer_pattern in self.target_layers):
                        count += 1
                else:
                    count += 1
        return count