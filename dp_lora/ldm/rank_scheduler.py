"""
Rank schedulers for dynamic rank scheduling during training.
Supports various scheduling strategies for LoRA/AdaLoRA rank adaptation.
"""

import numpy as np
from abc import ABC, abstractmethod


class RankScheduler(ABC):
    """Base class for rank schedulers."""
    
    def __init__(self, initial_rank: int, final_rank: int, total_steps: int):
        """
        Args:
            initial_rank: Starting rank at step 0
            final_rank: Target rank at final step
            total_steps: Total training steps
        """
        self.initial_rank = initial_rank
        self.final_rank = final_rank
        self.total_steps = total_steps
    
    @abstractmethod
    def get_rank(self, step: int) -> int:
        """Get rank for the given training step."""
        pass
    
    def __call__(self, step: int) -> int:
        return self.get_rank(step)


class LinearRankScheduler(RankScheduler):
    """Linearly interpolate rank from initial to final."""
    
    def get_rank(self, step: int) -> int:
        """Linear interpolation of rank."""
        t = min(step / self.total_steps, 1.0)
        rank = self.initial_rank + (self.final_rank - self.initial_rank) * t
        return max(1, int(round(rank)))


class CosineRankScheduler(RankScheduler):
    """Cosine annealing schedule for rank."""
    
    def get_rank(self, step: int) -> int:
        """Cosine annealing from initial to final rank."""
        t = min(step / self.total_steps, 1.0)
        rank = self.final_rank + 0.5 * (self.initial_rank - self.final_rank) * (1 + np.cos(np.pi * t))
        return max(1, int(round(rank)))


# Registry for rank schedulers
RANK_SCHEDULERS = {
    "linear": LinearRankScheduler,
    "cosine": CosineRankScheduler,
}


def get_rank_scheduler(name: str, **kwargs) -> RankScheduler:
    """
    Get a rank scheduler by name.
    
    Args:
        name: Scheduler name (one of the keys in RANK_SCHEDULERS)
        **kwargs: Arguments to pass to the scheduler
    
    Returns:
        Instantiated rank scheduler
    """
    if name not in RANK_SCHEDULERS:
        raise ValueError(f"Unknown rank scheduler: {name}. Available: {list(RANK_SCHEDULERS.keys())}")
    
    return RANK_SCHEDULERS[name](**kwargs)
