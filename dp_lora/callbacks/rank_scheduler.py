"""
PyTorch Lightning callback for dynamically scheduling LoRA rank during training.
"""

import logging
from typing import Optional, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from ldm.rank_scheduler import RankScheduler, get_rank_scheduler

logger = logging.getLogger(__name__)


class RankSchedulerCallback(Callback):
    """
    Callback to dynamically adjust LoRA rank during training based on a schedule.
    """
    
    def __init__(
        self,
        scheduler: Optional[RankScheduler] = None,
        scheduler_config: Optional[Dict] = None,
        adapter_name: str = "default",
        target_layers: Optional[list] = None,
        log_interval: int = 100,
        **kwargs
    ):
        """
        Args:
            scheduler: Pre-instantiated RankScheduler object
            scheduler_config: Dict with keys: name, initial_rank, final_rank, total_steps, and optional params
                Example: {"name": "cosine", "initial_rank": 16, "final_rank": 4, "total_steps": 10000}
            adapter_name: Name of the LoRA adapter to schedule
            target_layers: List of layer names to update. If None, updates all LoRA layers.
            log_interval: How often to log rank changes
        """
        super().__init__()
        
        self.adapter_name = adapter_name
        self.target_layers = target_layers
        self.log_interval = log_interval
        self.last_logged_rank = None
        
        # Initialize scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        elif scheduler_config is not None:
            self.scheduler = self._create_scheduler_from_config(scheduler_config)
        else:
            self.scheduler = None
            logger.warning("No scheduler provided to RankSchedulerCallback")
    
    def _create_scheduler_from_config(self, config: Dict) -> RankScheduler:
        """Create scheduler from config dictionary."""
        config = dict(config)  # Make a copy
        name = config.pop("name", "linear")
        return get_rank_scheduler(name, **config)
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when training starts."""
        if self.scheduler is None:
            logger.warning("RankSchedulerCallback: No scheduler configured, skipping rank scheduling")
            print("🚨 [RankScheduler] WARNING: No scheduler configured!")
            return
        
        msg = f"Starting rank scheduling with {self.scheduler.__class__.__name__}"
        logger.info(msg)
        print(f"✓ [RankScheduler] {msg}")
        
        msg = f"Initial rank: {self.scheduler.initial_rank}, Final rank: {self.scheduler.final_rank}, Total steps: {self.scheduler.total_steps}"
        logger.info(msg)
        print(f"✓ [RankScheduler] {msg}")
    
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """Called at the start of each training batch."""
        if self.scheduler is None:
            return
        
        global_step = trainer.global_step
        current_rank = self.scheduler.get_rank(global_step)
        
        # Update model ranks
        self._update_model_rank(pl_module, current_rank)
        
        # Log periodically or when rank changes
        if global_step % self.log_interval == 0 or current_rank != self.last_logged_rank:
            if current_rank != self.last_logged_rank:
                # Rank changed
                msg = f"Step {global_step}: LoRA rank changed to {current_rank} (prev: {self.last_logged_rank})"
                logger.info(msg)
                print(f"📊 [RankScheduler] {msg}")
                self.last_logged_rank = current_rank
            elif global_step % self.log_interval == 0:
                # Regular interval logging
                msg = f"Step {global_step}: LoRA rank = {current_rank}"
                logger.debug(msg)
                # Only print on major intervals (every 10x log_interval) to avoid spam
                if global_step % (self.log_interval * 10) == 0:
                    print(f"📊 [RankScheduler] {msg}")
    
    def _update_model_rank(self, pl_module: "pl.LightningModule", target_rank: int) -> None:
        """
        Update the rank of LoRA modules in the model.
        
        This method attempts to update rank for various model architectures:
        - Standard LoRA models
        - Diffusion models with LoRA
        """
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module
        
        updated = False
        
        # Try to find and update peft model
        if hasattr(model, 'peft_config'):
            self._update_peft_model_rank(model, target_rank)
            updated = True
        
        # Check for UNet with LoRA
        if hasattr(model, 'model') and hasattr(model.model, 'peft_config'):
            self._update_peft_model_rank(model.model, target_rank)
            updated = True
        
        # Update any modules with lora_A, lora_B parameters
        self._update_lora_parameters_rank(model, target_rank)
        
        if not updated:
            logger.warning("No PEFT model or LoRA layers found to update rank")
    
    def _update_peft_model_rank(self, model, target_rank: int) -> None:
        """Update rank in PEFT (Parameter-Efficient Fine-Tuning) models."""
        try:
            if hasattr(model, 'peft_config') and self.adapter_name in model.peft_config:
                config = model.peft_config[self.adapter_name]
                config.r = target_rank
        except Exception as e:
            logger.debug(f"Could not update PEFT model rank: {e}")
    
    def _update_lora_parameters_rank(self, model, target_rank: int) -> None:
        """
        Update rank-related metadata in LoRA layer parameters.
        This is a supplementary approach for models that directly store rank info.
        """
        try:
            for name, module in model.named_modules():
                # Skip if not a layer that might have rank info
                if not (hasattr(module, 'r') or hasattr(module, 'lora_A')):
                    continue
                
                # Filter by target layers if specified
                if self.target_layers is not None:
                    if not any(layer_pattern in name for layer_pattern in self.target_layers):
                        continue
                
                # Update rank dict if it exists (e.g., in LoraLayer)
                if hasattr(module, 'r') and isinstance(module.r, dict):
                    if self.adapter_name in module.r:
                        module.r[self.adapter_name] = target_rank
                
                # Update alpha dict based on new rank to maintain scaling
                if hasattr(module, 'alpha') and hasattr(module, 'scaling'):
                    if isinstance(module.alpha, dict) and self.adapter_name in module.alpha:
                        alpha = module.alpha[self.adapter_name]
                        # Update scaling: alpha / r
                        if isinstance(module.scaling, dict):
                            module.scaling[self.adapter_name] = alpha / target_rank
        except Exception as e:
            logger.debug(f"Error updating LoRA parameters rank: {e}")