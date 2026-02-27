"""
Dataset classes for longitudinal multimodal data.
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np


class LongitudinalDataset(Dataset):
    """
    Dataset for longitudinal multimodal clinical data.
    
    Supports multiple visits per patient and missing modalities.
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        visits_data: List[List[Dict[str, np.ndarray]]],
        missing_masks: Optional[List[List[Dict[str, int]]]] = None,
        min_completeness_ratio: Optional[float] = None,
        train_mask_rate: Optional[float] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            patient_ids: List of patient IDs.
            visits_data: List of patient visits.
                         Shape: (n_patients, n_visits, n_modalities).
                         Each element is a dict mapping modality names to numpy arrays.
            missing_masks: Optional pre-computed missing masks with values:
                           0 = available, 2 = naturally missing.
                           If None, masks will be inferred from None / NaN values.
            min_completeness_ratio: Optional filter threshold in [0, 1]. If set,
                           only keep patients whose ratio of available (visit, modality)
                           cells is >= this value.
            train_mask_rate: Optional active-mask rate in [0, 1]. If set, active
                           masking is applied per (visit, modality) on top of
                           natural missingness during __getitem__.
        """
        self.train_mask_rate = train_mask_rate
        self.min_completeness_ratio = min_completeness_ratio
        
        # Infer or use provided missing masks (natural missing: 2, available: 0)
        if missing_masks is None:
            base_missing_masks = self._infer_missing_masks(visits_data)
        else:
            base_missing_masks = missing_masks
        
        assert len(patient_ids) == len(visits_data)
        assert len(visits_data) == len(base_missing_masks)
        
        # Optionally filter patients by completeness ratio
        if min_completeness_ratio is not None:
            filtered_patient_ids: List[str] = []
            filtered_visits_data: List[List[Dict[str, np.ndarray]]] = []
            filtered_missing_masks: List[List[Dict[str, int]]] = []
            
            for pid, p_visits, p_masks in zip(patient_ids, visits_data, base_missing_masks):
                total_slots = 0
                available_slots = 0
                for visit_mask in p_masks:
                    for mask_value in visit_mask.values():
                        total_slots += 1
                        if mask_value == 0:
                            available_slots += 1
                if total_slots == 0:
                    completeness = 0.0
                else:
                    completeness = available_slots / float(total_slots)
                
                if completeness >= min_completeness_ratio:
                    filtered_patient_ids.append(pid)
                    filtered_visits_data.append(p_visits)
                    filtered_missing_masks.append(p_masks)
            
            self.patient_ids = filtered_patient_ids
            self.visits_data = filtered_visits_data
            self.base_missing_masks = filtered_missing_masks
        else:
            self.patient_ids = patient_ids
            self.visits_data = visits_data
            self.base_missing_masks = base_missing_masks
        
        # Validate
        assert len(self.patient_ids) == len(self.visits_data)
        assert len(self.visits_data) == len(self.base_missing_masks)
    
    def _infer_missing_masks(
        self,
        visits_data: List[List[Dict[str, np.ndarray]]]
    ) -> List[List[Dict[str, int]]]:
        """
        Infer missing masks from data (None or NaN values).
        
        Returns:
            List of per-patient, per-visit dictionaries with values:
            0 = available, 2 = naturally missing.
        """
        masks: List[List[Dict[str, int]]] = []
        for patient_visits in visits_data:
            patient_masks: List[Dict[str, int]] = []
            for visit in patient_visits:
                visit_mask: Dict[str, int] = {}
                for mod_name, mod_data in visit.items():
                    if mod_data is None:
                        visit_mask[mod_name] = 2  # Naturally missing
                    elif isinstance(mod_data, np.ndarray) and np.isnan(mod_data).all():
                        visit_mask[mod_name] = 2  # Naturally missing (all NaN)
                    else:
                        visit_mask[mod_name] = 0  # Available
                patient_masks.append(visit_mask)
            masks.append(patient_masks)
        return masks
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Tuple[
        str,
        List[Dict[str, torch.Tensor]],
        List[Dict[str, int]]
    ]:
        """
        Get a single patient's data.
        
        Returns:
            patient_id: Patient ID
            visits: List of visit data dictionaries (tensors)
            masks: List of integer masks per visit and modality
                   (0 = available, 1 = actively masked, 2 = naturally missing)
        """
        patient_id = self.patient_ids[idx]
        patient_visits = self.visits_data[idx]
        base_masks = self.base_missing_masks[idx]
        
        # Convert numpy arrays to tensors; keep None for naturally missing (mask == 2)
        visits_tensors: List[Dict[str, torch.Tensor]] = []
        for visit in patient_visits:
            visit_tensors: Dict[str, torch.Tensor] = {}
            for mod_name, mod_data in visit.items():
                if mod_data is not None:
                    visit_tensors[mod_name] = torch.as_tensor(mod_data, dtype=torch.float32)
                else:
                    visit_tensors[mod_name] = None
            visits_tensors.append(visit_tensors)
        
        # If no active masking, return natural masks (0/2 only)
        if self.train_mask_rate is None:
            return patient_id, visits_tensors, base_masks
        
        # Active-mask training: create 0/1/2 masks on top of natural missing (2)
        masks_012: List[Dict[str, int]] = []
        for visit_mask, visit_tensors in zip(base_masks, visits_tensors):
            new_visit_mask: Dict[str, int] = {}
            # First assign 0/1/2 based on natural missing and random active mask
            for mod_name, natural_value in visit_mask.items():
                if natural_value == 2:
                    # Naturally missing stays 2
                    new_visit_mask[mod_name] = 2
                else:
                    # Available: randomly choose 0 (available) or 1 (actively masked)
                    if np.random.rand() < float(self.train_mask_rate):
                        new_visit_mask[mod_name] = 1
                    else:
                        new_visit_mask[mod_name] = 0
            
            # Ensure at least one modality remains available (mask == 0) in this visit
            if all(mask_value != 0 for mask_value in new_visit_mask.values()):
                for mod_name, natural_value in visit_mask.items():
                    if natural_value == 0:
                        new_visit_mask[mod_name] = 0
                        break
            
            masks_012.append(new_visit_mask)
        
        return patient_id, visits_tensors, masks_012


def collate_visits(
    batch: List[Tuple[str, List[Dict[str, torch.Tensor]], List[Dict[str, int]]]]
) -> Tuple[
    List[str],
    List[List[Dict[str, torch.Tensor]]],
    List[List[Dict[str, int]]]
]:
    """
    Collate function for DataLoader.
    
    Since visits can have different lengths, we return lists instead of stacking.
    """
    patient_ids = []
    visits_list = []
    masks_list = []
    
    for patient_id, visits, masks in batch:
        patient_ids.append(patient_id)
        visits_list.append(visits)
        masks_list.append(masks)
    
    return patient_ids, visits_list, masks_list


def create_synthetic_data(
    n_patients: int = 100,
    n_visits: int = 3,
    modality_dims: Dict[str, int] = None,
    missing_rate: float = 0.2,
    seed: int = 42
) -> Tuple[
    List[str],
    List[List[Dict[str, np.ndarray]]],
    List[List[Dict[str, int]]]
]:
    """
    Create synthetic longitudinal data for testing.
    
    Args:
        n_patients: Number of patients
        n_visits: Number of visits per patient
        modality_dims: Dictionary of modality dimensions
        missing_rate: Probability of a modality being missing
        seed: Random seed
    
    Returns:
        patient_ids, visits_data, missing_masks
    """
    if modality_dims is None:
        modality_dims = {"lab": 61, "metab": 251}
    
    np.random.seed(seed)
    
    patient_ids = [f"P{i:04d}" for i in range(n_patients)]
    visits_data: List[List[Dict[str, np.ndarray]]] = []
    missing_masks: List[List[Dict[str, int]]] = []
    
    for p in range(n_patients):
        patient_visits: List[Dict[str, np.ndarray]] = []
        patient_masks: List[Dict[str, int]] = []
        
        for v in range(n_visits):
            visit_data: Dict[str, np.ndarray] = {}
            visit_mask: Dict[str, int] = {}
            
            for mod_name, mod_dim in modality_dims.items():
                # Randomly decide if this modality is missing
                is_missing = np.random.rand() < missing_rate
                
                if is_missing:
                    visit_data[mod_name] = None
                    visit_mask[mod_name] = 2  # Naturally missing
                else:
                    # Generate synthetic data
                    visit_data[mod_name] = np.random.randn(mod_dim).astype(np.float32)
                    visit_mask[mod_name] = 0  # Available
            
            patient_visits.append(visit_data)
            patient_masks.append(visit_mask)
        
        visits_data.append(patient_visits)
        missing_masks.append(patient_masks)
    
    return patient_ids, visits_data, missing_masks
