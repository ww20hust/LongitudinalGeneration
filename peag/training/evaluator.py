"""
Evaluator for PEAG framework.

This module implements test set evaluation and ablation study evaluation
with metrics: Pearson correlation coefficient, MAE, and MSE.

Reference: Evaluation task description
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from peag.model import PEAGModel
from peag.utils.metrics import compute_all_metrics


class Evaluator:
    """
    Evaluator class for PEAG model.
    
    Implements:
    - Test set evaluation
    - Ablation study evaluation (full, baseline_only, followup_only modes)
    - Metric computation (Pearson r, MAE, MSE)
    """
    
    def __init__(
        self,
        model: PEAGModel,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PEAGModel instance
            device: Device to run evaluation on
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        mode: str = "full",
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluate model on test set.
        
        Reference: Evaluation task description
        
        Args:
            test_loader: DataLoader for test dataset
            mode: Ablation mode - "full", "baseline_only", or "followup_only"
            return_predictions: Whether to return individual predictions
        
        Returns:
            Dictionary containing:
            - metrics: Evaluation metrics (Pearson r, MAE, MSE)
            - predictions: Individual predictions (if return_predictions=True)
        """
        all_pred_metab = []
        all_true_metab = []
        
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating ({mode})"):
                baseline_lab = batch["baseline_lab_tests"].to(self.device)
                baseline_metab = batch["baseline_metabolomics"].to(self.device)
                followup_lab = batch["followup_lab_tests"].to(self.device)
                followup_metab = batch.get("followup_metabolomics")
                
                if followup_metab is not None:
                    followup_metab = followup_metab.to(self.device)
                
                # Forward pass
                output = self.model(
                    baseline_lab, baseline_metab,
                    followup_lab, followup_metab,
                    mode=mode,
                    kl_annealing_weight=1.0
                )
                
                # Get predictions
                pred_metab = output["metab_recon_followup"].cpu().numpy()
                
                # Get ground truth (use followup if available, otherwise baseline)
                if followup_metab is not None:
                    true_metab = followup_metab.cpu().numpy()
                else:
                    true_metab = baseline_metab.cpu().numpy()
                
                all_pred_metab.append(pred_metab)
                all_true_metab.append(true_metab)
                
                if return_predictions:
                    predictions.append({
                        "predicted": pred_metab,
                        "true": true_metab
                    })
        
        # Concatenate all predictions
        all_pred_metab = np.concatenate(all_pred_metab, axis=0)
        all_true_metab = np.concatenate(all_true_metab, axis=0)
        
        # Compute metrics
        metrics = compute_all_metrics(all_true_metab, all_pred_metab)
        
        result = {
            "mode": mode,
            "metrics": metrics,
            "n_samples": len(all_true_metab)
        }
        
        if return_predictions:
            result["predictions"] = predictions
        
        return result
    
    def evaluate_ablation_study(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluate all ablation modes and compare performance.
        
        Reference: Ablation study description
        
        Args:
            test_loader: DataLoader for test dataset
            return_predictions: Whether to return individual predictions
        
        Returns:
            Dictionary containing:
            - full: Results for full PEAG mode
            - baseline_only: Results for baseline-only mode
            - followup_only: Results for follow-up-only mode
            - comparison: Summary comparison
        """
        print("Evaluating full PEAG mode...")
        full_results = self.evaluate(
            test_loader,
            mode="full",
            return_predictions=return_predictions
        )
        
        print("Evaluating baseline-only mode...")
        baseline_results = self.evaluate(
            test_loader,
            mode="baseline_only",
            return_predictions=return_predictions
        )
        
        print("Evaluating follow-up-only mode...")
        followup_results = self.evaluate(
            test_loader,
            mode="followup_only",
            return_predictions=return_predictions
        )
        
        # Create comparison
        comparison = {
            "full": full_results["metrics"],
            "baseline_only": baseline_results["metrics"],
            "followup_only": followup_results["metrics"]
        }
        
        results = {
            "full": full_results,
            "baseline_only": baseline_results,
            "followup_only": followup_results,
            "comparison": comparison
        }
        
        return results
    
    def print_results(self, results: dict):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Results dictionary from evaluate or evaluate_ablation_study
        """
        if "comparison" in results:
            # Ablation study results
            print("\n" + "="*60)
            print("Ablation Study Results")
            print("="*60)
            
            for mode, metrics in results["comparison"].items():
                print(f"\n{mode.upper().replace('_', ' ')}:")
                print(f"  Pearson r: {metrics['pearson_r']:.4f}")
                print(f"  MAE:       {metrics['mae']:.6f}")
                print(f"  MSE:       {metrics['mse']:.6f}")
            
            print("\n" + "="*60)
        else:
            # Single mode results
            print("\n" + "="*60)
            print(f"Evaluation Results ({results['mode']})")
            print("="*60)
            metrics = results["metrics"]
            print(f"Pearson r: {metrics['pearson_r']:.4f}")
            print(f"MAE:       {metrics['mae']:.6f}")
            print(f"MSE:       {metrics['mse']:.6f}")
            print(f"N samples: {results['n_samples']}")
            print("="*60)

