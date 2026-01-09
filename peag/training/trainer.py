"""
Trainer for PEAG framework.

This module implements the training loop with ELBO optimization, KL annealing,
early stopping, and checkpoint management.

Reference: Methods section - Training Implementation Details
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from peag.model import PEAGModel
from peag.training.config import TrainingConfig
from peag.losses import adversarial_loss_discriminator


class Trainer:
    """
    Trainer class for PEAG model.
    
    Implements:
    - ELBO optimization
    - KL annealing (linear increase from 0 to 1 over first 50 epochs)
    - Early stopping (stops if validation loss doesn't improve for 50 epochs)
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: PEAGModel,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset = None,
        config: TrainingConfig = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PEAGModel instance
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional, will split from train if None)
            config: Training configuration (uses default if None)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        
        # Move model to device
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.val_loader = None
        
        # Optimizers
        # Main model optimizer (encoders, decoders)
        model_params = [
            p for n, p in self.model.named_parameters()
            if "discriminator" not in n
        ]
        self.optimizer = optim.AdamW(
            model_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Discriminator optimizer
        disc_params = [
            p for n, p in self.model.named_parameters()
            if "discriminator" in n
        ]
        self.disc_optimizer = optim.AdamW(
            disc_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "val_recon_loss": []
        }
        
        # Create checkpoint directory
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def compute_kl_annealing_weight(self, epoch: int) -> float:
        """
        Compute KL annealing weight for current epoch.
        
        Linear increase from 0 to 1 over first kl_annealing_epochs epochs.
        
        Reference: Methods section - Training Implementation Details
        
        Args:
            epoch: Current epoch number
        
        Returns:
            KL annealing weight (0 to 1)
        """
        if epoch < self.config.kl_annealing_epochs:
            return epoch / self.config.kl_annealing_epochs
        else:
            return 1.0
    
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        
        kl_weight = self.compute_kl_annealing_weight(epoch)
        
        total_losses = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "alignment_loss": 0.0,
            "adversarial_loss": 0.0,
            "discriminator_loss": 0.0
        }
        
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
        
        for batch in pbar:
            # Move batch to device
            baseline_lab = batch["baseline_lab_tests"].to(self.device)
            baseline_metab = batch["baseline_metabolomics"].to(self.device)
            followup_lab = batch["followup_lab_tests"].to(self.device)
            followup_metab = batch.get("followup_metabolomics")
            if followup_metab is not None:
                followup_metab = followup_metab.to(self.device)
            
            # ===== TRAIN DISCRIMINATOR =====
            self.disc_optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                output = self.model(
                    baseline_lab, baseline_metab,
                    followup_lab, followup_metab,
                    mode="full",
                    kl_annealing_weight=kl_weight
                )
            
            # Discriminator loss
            real_metab = followup_metab if followup_metab is not None else baseline_metab
            fake_metab = output["metab_recon_followup"]
            
            disc_loss = adversarial_loss_discriminator(
                self.model.discriminator,
                real_metab,
                fake_metab.detach()
            )
            
            disc_loss.backward()
            self.disc_optimizer.step()
            
            # ===== TRAIN MAIN MODEL =====
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(
                baseline_lab, baseline_metab,
                followup_lab, followup_metab,
                mode="full",
                kl_annealing_weight=kl_weight
            )
            
            losses = output["losses"]
            total_loss = losses["total_loss"]
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_losses["total_loss"] += total_loss.item()
            total_losses["reconstruction_loss"] += losses["reconstruction_loss"].item()
            total_losses["kl_loss"] += losses["kl_loss"].item()
            total_losses["alignment_loss"] += losses["alignment_loss"].item()
            total_losses["adversarial_loss"] += losses["adversarial_loss"].item()
            total_losses["discriminator_loss"] += disc_loss.item()
            
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss.item(),
                "recon": losses["reconstruction_loss"].item(),
                "kl": losses["kl_loss"].item()
            })
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def validate(self) -> dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of average validation losses
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_losses = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "alignment_loss": 0.0,
            "adversarial_loss": 0.0
        }
        
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                baseline_lab = batch["baseline_lab_tests"].to(self.device)
                baseline_metab = batch["baseline_metabolomics"].to(self.device)
                followup_lab = batch["followup_lab_tests"].to(self.device)
                followup_metab = batch.get("followup_metabolomics")
                if followup_metab is not None:
                    followup_metab = followup_metab.to(self.device)
                
                output = self.model(
                    baseline_lab, baseline_metab,
                    followup_lab, followup_metab,
                    mode="full",
                    kl_annealing_weight=1.0  # No annealing for validation
                )
                
                losses = output["losses"]
                
                total_losses["total_loss"] += losses["total_loss"].item()
                total_losses["reconstruction_loss"] += losses["reconstruction_loss"].item()
                total_losses["kl_loss"] += losses["kl_loss"].item()
                total_losses["alignment_loss"] += losses["alignment_loss"].item()
                total_losses["adversarial_loss"] += losses["adversarial_loss"].item()
                
                n_batches += 1
        
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.config.save_dir, "latest.pt"))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.save_dir, "best.pt"))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]
    
    def train(self):
        """
        Main training loop with early stopping.
        
        Reference: Methods section - Training Implementation Details
        """
        print(f"Starting training on device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_loader is not None:
            print(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            
            # Update history
            self.training_history["train_loss"].append(train_losses["total_loss"])
            self.training_history["train_recon_loss"].append(train_losses["reconstruction_loss"])
            
            if val_losses:
                self.training_history["val_loss"].append(val_losses["total_loss"])
                self.training_history["val_recon_loss"].append(val_losses["reconstruction_loss"])
                
                val_loss = val_losses["reconstruction_loss"]  # Use reconstruction loss for early stopping
                
                # Check for improvement
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                if self.config.save_best:
                    self.save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation loss: {self.best_val_loss:.6f}")
                    break
            else:
                # No validation set, just save latest
                if self.config.save_best:
                    self.save_checkpoint(epoch, is_best=False)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.6f}")
            print(f"Train Recon: {train_losses['reconstruction_loss']:.6f}")
            if val_losses:
                print(f"Val Loss: {val_losses['total_loss']:.6f}")
                print(f"Val Recon: {val_losses['reconstruction_loss']:.6f}")
                print(f"Best Val Loss: {self.best_val_loss:.6f}")
                print(f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
        
        print("\nTraining completed!")
        
        # Load best model
        if self.config.save_best and self.val_loader is not None:
            best_path = os.path.join(self.config.save_dir, "best.pt")
            if os.path.exists(best_path):
                print(f"Loading best model from {best_path}")
                self.load_checkpoint(best_path)

