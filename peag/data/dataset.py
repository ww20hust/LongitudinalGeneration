"""
PyTorch Dataset class for longitudinal clinical data.

This module implements a Dataset class for handling paired Baseline and Follow-up
visit data with support for different access modes.

Reference: Data format specification
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class LongitudinalDataset(Dataset):
    """
    Dataset class for longitudinal clinical visits.
    
    Stores paired Baseline and Follow-up visit data and supports different
    access modes (full, lab tests only, etc.).
    """
    
    def __init__(
        self,
        baseline_lab_tests: np.ndarray,
        baseline_metabolomics: np.ndarray,
        followup_lab_tests: np.ndarray,
        followup_metabolomics: np.ndarray = None,
        normalize: bool = True
    ):
        """
        Initialize longitudinal dataset.
        
        Args:
            baseline_lab_tests: Baseline lab tests of shape (n_samples, 61)
            baseline_metabolomics: Baseline metabolomics of shape (n_samples, 251)
            followup_lab_tests: Follow-up lab tests of shape (n_samples, 61)
            followup_metabolomics: Follow-up metabolomics of shape (n_samples, 251) or None
            normalize: Whether to normalize data (default: True)
        """
        # Convert to numpy if needed
        self.baseline_lab_tests = np.asarray(baseline_lab_tests, dtype=np.float32)
        self.baseline_metabolomics = np.asarray(baseline_metabolomics, dtype=np.float32)
        self.followup_lab_tests = np.asarray(followup_lab_tests, dtype=np.float32)
        
        if followup_metabolomics is not None:
            self.followup_metabolomics = np.asarray(followup_metabolomics, dtype=np.float32)
        else:
            self.followup_metabolomics = None
        
        self.n_samples = self.baseline_lab_tests.shape[0]
        
        # Normalize data
        if normalize:
            self._normalize()
    
    def _normalize(self):
        """Normalize data using mean and std."""
        # Compute statistics from baseline data
        self.lab_mean = np.mean(self.baseline_lab_tests, axis=0, keepdims=True)
        self.lab_std = np.std(self.baseline_lab_tests, axis=0, keepdims=True) + 1e-8
        self.metab_mean = np.mean(self.baseline_metabolomics, axis=0, keepdims=True)
        self.metab_std = np.std(self.baseline_metabolomics, axis=0, keepdims=True) + 1e-8
        
        # Normalize baseline
        self.baseline_lab_tests = (self.baseline_lab_tests - self.lab_mean) / self.lab_std
        self.baseline_metabolomics = (self.baseline_metabolomics - self.metab_mean) / self.metab_std
        
        # Normalize follow-up
        self.followup_lab_tests = (self.followup_lab_tests - self.lab_mean) / self.lab_std
        if self.followup_metabolomics is not None:
            self.followup_metabolomics = (self.followup_metabolomics - self.metab_mean) / self.metab_std
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary containing:
            - baseline_lab_tests: Baseline lab tests
            - baseline_metabolomics: Baseline metabolomics
            - followup_lab_tests: Follow-up lab tests
            - followup_metabolomics: Follow-up metabolomics (if available)
        """
        sample = {
            "baseline_lab_tests": torch.FloatTensor(self.baseline_lab_tests[idx]),
            "baseline_metabolomics": torch.FloatTensor(self.baseline_metabolomics[idx]),
            "followup_lab_tests": torch.FloatTensor(self.followup_lab_tests[idx]),
        }
        
        if self.followup_metabolomics is not None:
            sample["followup_metabolomics"] = torch.FloatTensor(self.followup_metabolomics[idx])
        
        return sample

