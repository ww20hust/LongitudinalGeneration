"""
Data loaders for UK Biobank format data.

This module implements data loading and preprocessing for paired Baseline and
Follow-up visit data with 70/30 train-test split.

Reference: Benchmarking task description
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from peag.data.dataset import LongitudinalDataset


class UKBiobankLoader:
    """
    Data loader for UK Biobank format longitudinal data.
    
    Loads paired data from Baseline and Follow-up visits and supports
    70/30 train-test split.
    """
    
    def __init__(
        self,
        baseline_lab_tests: np.ndarray,
        baseline_metabolomics: np.ndarray,
        followup_lab_tests: np.ndarray,
        followup_metabolomics: np.ndarray = None,
        test_size: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize UK Biobank data loader.
        
        Args:
            baseline_lab_tests: Baseline lab tests of shape (n_samples, 61)
            baseline_metabolomics: Baseline metabolomics of shape (n_samples, 251)
            followup_lab_tests: Follow-up lab tests of shape (n_samples, 61)
            followup_metabolomics: Follow-up metabolomics of shape (n_samples, 251) or None
            test_size: Proportion of data for test set (default: 0.3)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.baseline_lab_tests = np.asarray(baseline_lab_tests, dtype=np.float32)
        self.baseline_metabolomics = np.asarray(baseline_metabolomics, dtype=np.float32)
        self.followup_lab_tests = np.asarray(followup_lab_tests, dtype=np.float32)
        
        if followup_metabolomics is not None:
            self.followup_metabolomics = np.asarray(followup_metabolomics, dtype=np.float32)
        else:
            self.followup_metabolomics = None
        
        self.test_size = test_size
        self.random_state = random_state
        
        # Verify dimensions
        n_samples = self.baseline_lab_tests.shape[0]
        assert self.baseline_metabolomics.shape[0] == n_samples
        assert self.followup_lab_tests.shape[0] == n_samples
        assert self.baseline_lab_tests.shape[1] == 61
        assert self.baseline_metabolomics.shape[1] == 251
        assert self.followup_lab_tests.shape[1] == 61
        
        if self.followup_metabolomics is not None:
            assert self.followup_metabolomics.shape[0] == n_samples
            assert self.followup_metabolomics.shape[1] == 251
    
    def split_data(self) -> tuple:
        """
        Split data into training and test sets (70/30).
        
        Returns:
            Tuple of (train_data, test_data) where each contains:
            - baseline_lab_tests
            - baseline_metabolomics
            - followup_lab_tests
            - followup_metabolomics (if available)
        """
        n_samples = self.baseline_lab_tests.shape[0]
        indices = np.arange(n_samples)
        
        train_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Split baseline data
        train_baseline_lab = self.baseline_lab_tests[train_indices]
        train_baseline_metab = self.baseline_metabolomics[train_indices]
        test_baseline_lab = self.baseline_lab_tests[test_indices]
        test_baseline_metab = self.baseline_metabolomics[test_indices]
        
        # Split follow-up data
        train_followup_lab = self.followup_lab_tests[train_indices]
        test_followup_lab = self.followup_lab_tests[test_indices]
        
        train_followup_metab = None
        test_followup_metab = None
        if self.followup_metabolomics is not None:
            train_followup_metab = self.followup_metabolomics[train_indices]
            test_followup_metab = self.followup_metabolomics[test_indices]
        
        train_data = {
            "baseline_lab_tests": train_baseline_lab,
            "baseline_metabolomics": train_baseline_metab,
            "followup_lab_tests": train_followup_lab,
            "followup_metabolomics": train_followup_metab
        }
        
        test_data = {
            "baseline_lab_tests": test_baseline_lab,
            "baseline_metabolomics": test_baseline_metab,
            "followup_lab_tests": test_followup_lab,
            "followup_metabolomics": test_followup_metab
        }
        
        return train_data, test_data
    
    def get_datasets(
        self,
        normalize: bool = True
    ) -> tuple[LongitudinalDataset, LongitudinalDataset]:
        """
        Get training and test datasets.
        
        Args:
            normalize: Whether to normalize data (default: True)
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_data, test_data = self.split_data()
        
        train_dataset = LongitudinalDataset(
            baseline_lab_tests=train_data["baseline_lab_tests"],
            baseline_metabolomics=train_data["baseline_metabolomics"],
            followup_lab_tests=train_data["followup_lab_tests"],
            followup_metabolomics=train_data["followup_metabolomics"],
            normalize=normalize
        )
        
        test_dataset = LongitudinalDataset(
            baseline_lab_tests=test_data["baseline_lab_tests"],
            baseline_metabolomics=test_data["baseline_metabolomics"],
            followup_lab_tests=test_data["followup_lab_tests"],
            followup_metabolomics=test_data["followup_metabolomics"],
            normalize=normalize
        )
        
        return train_dataset, test_dataset

