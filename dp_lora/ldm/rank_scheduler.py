"""
Layer-wise rank scheduler for LoRA fine-tuning.
Assigns fixed ranks to layers based on their depth in the model.
"""

import numpy as np


class LayerWiseRankScheduler:
    """
    Fixed rank scheduler that assigns different ranks to different layers based on depth.
    Does NOT schedule ranks over time - each layer gets a fixed rank based on its position.
    """
    
    def __init__(
        self, 
        initial_rank: int, 
        final_rank: int, 
        total_steps: int = None,
        schedule_type: str = "linear",
        num_layers: int = None,
        **kwargs
    ):
        """
        Args:
            initial_rank: Rank for the first layers (early in the model)
            final_rank: Rank for the last layers (late in the model)
            total_steps: Ignored (for config compatibility)
            schedule_type: "linear" or "cosine" - how to interpolate ranks across layers
            num_layers: Total number of layers in the model (set during callback init)
        """
        self.initial_rank = initial_rank
        self.final_rank = final_rank
        self.schedule_type = schedule_type.lower()
        self.num_layers = num_layers
        self.layer_ranks = {}  # Cache: layer_index -> rank
        
        if self.schedule_type not in ["linear", "cosine"]:
            raise ValueError(f"schedule_type must be 'linear' or 'cosine', got {schedule_type}")
    
    def set_num_layers(self, num_layers: int) -> None:
        """Set the number of layers after the model is initialized."""
        self.num_layers = num_layers
        self.layer_ranks = {}  # Reset cache
    
    def get_rank_for_layer(self, layer_index: int) -> int:
        """
        Get the fixed rank for a specific layer based on its index.
        Layer 0 is the first layer (gets initial_rank or close to it).
        Layer num_layers-1 is the last layer (gets final_rank).
        
        Args:
            layer_index: 0-indexed position of the layer in the model
            
        Returns:
            Fixed rank for this layer
        """
        if self.num_layers is None:
            raise RuntimeError("num_layers not set. Call set_num_layers() first.")
        
        if layer_index in self.layer_ranks:
            return self.layer_ranks[layer_index]
        
        if self.num_layers == 1:
            rank = self.initial_rank
        else:
            # Normalize layer position to [0, 1]
            t = layer_index / (self.num_layers - 1)
            
            if self.schedule_type == "linear":
                # Linear interpolation
                rank = self.initial_rank + (self.final_rank - self.initial_rank) * t
            else:  # cosine
                # Cosine annealing: start high, decay to final_rank
                rank = self.final_rank + 0.5 * (self.initial_rank - self.final_rank) * (1 + np.cos(np.pi * t))
        
        rank = max(1, int(round(rank)))
        self.layer_ranks[layer_index] = rank
        return rank


# Registry for rank schedulers
RANK_SCHEDULERS = {
    "layer-wise": LayerWiseRankScheduler,
}


def get_rank_scheduler(name: str, **kwargs) -> LayerWiseRankScheduler:
    """
    Get a rank scheduler by name.
    
    Args:
        name: Scheduler name (currently only "layer-wise" is supported)
        **kwargs: Arguments to pass to the scheduler
    
    Returns:
        Instantiated rank scheduler
    """
    if name not in RANK_SCHEDULERS:
        raise ValueError(f"Unknown rank scheduler: {name}. Available: {list(RANK_SCHEDULERS.keys())}")
    
    return RANK_SCHEDULERS[name](**kwargs)
