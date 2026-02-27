"""
Training utilities for PEAG framework.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import os
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class Trainer:
    """
    Trainer for PEAG model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        kl_anneal_epochs: int = 50
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.kl_anneal_epochs = kl_anneal_epochs
        self.current_epoch = 0
    
    def compute_kl_annealing_weight(self) -> float:
        """Compute KL annealing weight (linear increase from 0 to 1)."""
        if self.kl_anneal_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.kl_anneal_epochs)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "alignment_loss": 0.0,
            "adversarial_loss": 0.0
        }
        num_batches = 0
        kl_weight = self.compute_kl_annealing_weight()
        use_active_mask = getattr(dataloader.dataset, "train_mask_rate", None) is not None
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch}"):
            patient_ids, visits_list, masks_list = batch
            
            # Move data to device
            visits_list_device = []
            for visits in visits_list:
                visits_device = []
                for visit in visits:
                    visit_device = {
                        k: v.to(self.device) if v is not None else None
                        for k, v in visit.items()
                    }
                    visits_device.append(visit_device)
                visits_list_device.append(visits_device)
            
            # Forward pass for each patient
            batch_losses = []
            batch_outputs = []
            for visits, masks in zip(visits_list_device, masks_list):
                if use_active_mask:
                    # Build masked input: None where mask is 1 (actively masked) or 2 (naturally missing)
                    visits_masked: List[Dict[str, torch.Tensor]] = []
                    for visit_data, visit_mask in zip(visits, masks):
                        masked_visit: Dict[str, torch.Tensor] = {}
                        for mod_name, tensor in visit_data.items():
                            mask_value = visit_mask.get(mod_name, 2)
                            if mask_value in (1, 2):
                                masked_visit[mod_name] = None
                            else:
                                masked_visit[mod_name] = tensor
                        visits_masked.append(masked_visit)
                    
                    output = self.model(
                        visits_data=visits_masked,
                        missing_masks=masks,
                        kl_annealing_weight=kl_weight,
                        recon_targets=visits,
                    )
                else:
                    output = self.model(
                        visits_data=visits,
                        missing_masks=masks,
                        kl_annealing_weight=kl_weight,
                    )
                batch_losses.append(output["losses"]["total_loss"])
                batch_outputs.append(output)
            
            # Average loss over batch
            loss = torch.stack(batch_losses).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses from this batch
            for output in batch_outputs:
                for key in epoch_losses.keys():
                    if key in output["losses"]:
                        epoch_losses[key] += output["losses"][key].item()
            
            num_batches += len(batch_outputs)
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        self.current_epoch += 1
        return epoch_losses
    
    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        save_dir: Optional[str] = None,
        validate_every: int = 10
    ) -> List[Dict[str, float]]:
        """
        Train the model.
        
        Args:
            dataloader: Training data loader
            n_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            validate_every: Validate every N epochs
        
        Returns:
            List of loss dictionaries per epoch
        """
        history = []
        
        for epoch in range(n_epochs):
            epoch_losses = self.train_epoch(dataloader)
            history.append(epoch_losses)
            
            print(f"Epoch {epoch + 1}/{n_epochs}")
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.4f}")
            
            # Save checkpoint
            if save_dir is not None and (epoch + 1) % validate_every == 0:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path)
        
        return history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {path}")
