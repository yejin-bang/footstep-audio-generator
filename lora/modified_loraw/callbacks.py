import os
import torch
import pytorch_lightning as pl
from weakref import proxy

from .network import LoRAWrapper

# Save lora weights only
class LoRAModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, lora: LoRAWrapper, **kwargs):
        super().__init__(**kwargs)
        self.lora = lora

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.lora.save_weights(filepath)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


# Update and save base model
class ReLoRAModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, lora: LoRAWrapper, checkpoint_every_n_updates=1, **kwargs):
        super().__init__(**kwargs)
        self.lora = lora
        self.checkpoint_every_n_updates = checkpoint_every_n_updates
        self.updates = 0

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        self.lora.net.update_base()

        if self.updates % self.checkpoint_every_n_updates == 0:
            super()._save_checkpoint(trainer, filepath)

        self.updates += 1


# Update base model with lora weights (no checkpoint saving)
class ReLoRAUpdateCallback(pl.Callback):

    def __init__(self, lora: LoRAWrapper, update_every=1000, **kwargs):
        super().__init__(**kwargs)
        self.lora = lora
        self.update_every = update_every

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        
        if (trainer.global_step - 1) % self.update_every != 0:
            return

        self.lora.net.update_base()

class GradientMonitorCallback(pl.Callback):
    def __init__(self, lora=None, log_every_n_steps=1):
        super().__init__()
        self.lora = lora
        self.log_every_n_steps = log_every_n_steps
        self.gradient_norms = []
        self.clipped_count = 0
        self.total_count = 0
        
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.lora is None:
            return
            
        # Calculate gradient norm for LoRA parameters ONLY
        total_norm = 0.0
        for p in self.lora.residual_modules.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Check if this gradient will be clipped
        clip_val = trainer.gradient_clip_val
        is_clipped = total_norm > clip_val if clip_val > 0 else False
        
        # Track statistics
        self.gradient_norms.append(total_norm)
        self.total_count += 1
        if is_clipped:
            self.clipped_count += 1
        
        # Log every N steps
        if trainer.global_step % self.log_every_n_steps == 0:
            clip_status = "CLIPPED" if is_clipped else "OK"
            print(f"Step {trainer.global_step:5d} | LoRA Grad Norm: {total_norm:.4f} | Status: {clip_status}")
        
        # Print summary every 100 steps
        if trainer.global_step > 0 and trainer.global_step % 100 == 0:
            clip_pct = (self.clipped_count / self.total_count) * 100
            avg_norm = sum(self.gradient_norms[-100:]) / min(100, len(self.gradient_norms))
            print(f"\n{'='*60}")
            print(f"LoRA Gradient Summary {self.log_every_n_steps}:")
            print(f"  Average LoRA gradient norm: {avg_norm:.4f}")
            print(f"  Clipped: {self.clipped_count}/{self.total_count} steps ({clip_pct:.1f}%)")
            print(f"  Clip threshold: {clip_val}")
            print(f"{'='*60}\n")