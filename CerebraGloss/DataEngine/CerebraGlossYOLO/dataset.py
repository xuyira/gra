import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Tuple
from copy import deepcopy


class FPN_dataset(Dataset):
    def __init__(self, 
                 S_LEVELS: List[int], 
                 anchors_per_level: List[List[float]], 
                 folders: Optional[List[str]] = None, 
                 classes: Optional[List[str]] = None,
                 use_augmentation: bool = False, # NOTE Always False when inferencing
                 do_zscore: bool = True):
        """
        Dataset class for preparing data for FPN models.

        Args:
            S_LEVELS (List[int]): List of grid numbers for each FPN prediction level. E.g. [250, 32] for P3, P6
            anchors_per_level (List[List[float]]): List of anchor widths for each FPN level (normalized). E.g. [[0.06], [0.95]] for P3, P6
            folders (Optional[List[str]]): Dataset folders for loading data.
            classes (Optional[List[str]]): List of label names.
            use_augmentation (bool): Whether to use data augmentation.
            do_zscore (bool): Whether to apply per-channel zscore normalization to EEG data, otherwise use raw microvolt data (for visualizing raw data)
        """
        # --- Basic settings ---
        ROOT_DIR = '.'
        self.folders = folders if folders is not None else [f'TUseedCorpus_{i}' for i in range(1, 7)]
        self.sampling_rate = 200
        self.seq_len = 2000  # 200Hz * 10s
        self.do_zscore = do_zscore

        # --- FPN specific settings ---
        self.S_LEVELS = S_LEVELS
        self.anchors_per_level = anchors_per_level
        assert len(self.S_LEVELS) == len(self.anchors_per_level), \
            "Number of grid levels must match number of anchor levels"
        self.num_levels = len(self.S_LEVELS)

        # --- Class settings ---
        if classes is None:
            self.classes = ['sharp','spike','spsw','alpha','delta','spindle','Kcomplex','eyem','eyer+','eyer-','hfnoise']
        else:
            self.classes = classes
        self.class2idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # --- Channel normalization settings ---
        self.num_std_channels = 19
        self.STD_19_CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']

        # --- Load data samples ---
        self.samples = self._load_samples(ROOT_DIR)

    def _load_samples(self, root_dir: str) -> List[Dict[str, Any]]:
        """Load sample information from files."""
        samples = []
        for folder in self.folders:
            folder_path = os.path.join(root_dir, folder)
            json_path = os.path.join(folder_path, 'annotation.json')

            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            npy_filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

            for npy_filename in npy_filenames:
                key = npy_filename[:-7] # Remove '.npy' and channel suffix
                if key not in annotations: continue

                npy_path = os.path.join(folder_path, npy_filename)
                bboxes = annotations[key].get("bboxes", [])
                ref_type = 'ear_ref' if npy_filename.endswith('_er.npy') else 'ave_ref'
                ch_names = annotations[key]['channels'][ref_type]
                ch_names = [ch_name.split('-')[0] for ch_name in ch_names] # Keep only channel name, remove possible reference method

                # Ensure at least one annotation of the target class exists in the sample
                # has_target_bbox = any(bbox.get("wave_type") in self.classes for bbox in bboxes)
                # if has_target_bbox:
                samples.append({ # Still put all samples in, regardless of whether the sample has annotation of the target class
                    "npy_path": npy_path,
                    "bboxes": bboxes,
                    "ch_names": ch_names
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[Tensor, List[Tensor], Tensor, Tensor]:
                - eeg_tensor (Tensor): (19, 2000) EEG data tensor after padding and normalization
                - target_tensors (List[Tensor]): list of FPN, in format of (C, S_level, B_level, 3 + num_classes)
                - attention_mask (Tensor): (19,) mask for padded channels, True for padded channels
                - pos_indices (Tensor): (19,) index of channel positions in standard 10-20 arrangement
        """
        sample = self.samples[index]
        # Use deepcopy to create an independent copy of bboxes to avoid modifying original data
        bboxes = deepcopy(sample["bboxes"])

        # 1. Load and normalize EEG data
        eeg_data = np.load(sample["npy_path"]) # EEG (C,2000)
        ch_names = sample["ch_names"] # Corresponding channel names (C,)
        assert eeg_data.shape[0] == len(ch_names), "Number of EEG channels and names do not match"

        attention_mask = torch.tensor([ch not in ch_names for ch in self.STD_19_CHANNELS], dtype=torch.bool)
        # pos_indices[i] represents the position of the i-th channel in eeg_tensor according to the standard 10-20 arrangement. Since our EEG padding scheme follows the standard electrode layout, pos_indices can be initialized as below
        pos_indices = torch.arange(self.num_std_channels, dtype=torch.long)
        padded_eeg_data = np.zeros((self.num_std_channels, self.seq_len), dtype=np.float32) # (19,2000)
        ch_idx_to_std_idx = [self.STD_19_CHANNELS.index(ch_name) for ch_name in ch_names] # (C,) Channel names of current sample, corresponding to positions in standard arrangement

        # Pad (C,2000) to standard 10-20 arrangement, (19,2000)
        for i in range(len(ch_names)):
            padded_eeg_data[ch_idx_to_std_idx[i]] = eeg_data[i]
        eeg_tensor = torch.from_numpy(padded_eeg_data)

        # zscore normalization
        if self.do_zscore:
            mean = eeg_tensor.mean(dim=1, keepdim=True)
            std = eeg_tensor.std(dim=1, keepdim=True)
            eeg_tensor = (eeg_tensor - mean) / (std + 1e-6) # Add epsilon to avoid division by zero

        # Modify bounding boxes, the original index is the index of ch_names, needs to be changed after padding
        # Also unify bbox["channel_idx"] to List[int]
        for bbox in bboxes:
            if isinstance(bbox["channel_idx"],int):
                bbox["channel_idx"] = [ch_idx_to_std_idx[bbox["channel_idx"]]]
            else: # List[int]
                bbox["channel_idx"] = [ch_idx_to_std_idx[ch_idx] for ch_idx in bbox["channel_idx"]]

        # 2. Apply data augmentation (if enabled)
        # if self.use_augmentation:
        #     eeg_tensor, bboxes, attention_mask, pos_indices = self.augmentation_pipeline(
        #         eeg_tensor, bboxes, attention_mask, pos_indices
        #     )

        # If random channel transformation is performed, bbox channel_idx needs to be mapped to the new shuffled channel position via original_to_current_map[channel_idx]
        # e.g. [3,2,0,1][2,3,1,0] = [0,1,2,3]
        original_to_current_map = torch.zeros_like(pos_indices)
        original_to_current_map[pos_indices] = torch.arange(self.num_std_channels)

        # 3. Create target tensor for each FPN level
        target_tensors = []
        for i in range(self.num_levels): # Iterate over FPN feature map levels
            S = self.S_LEVELS[i] # Number of grids in this level
            num_anchors_in_level = len(self.anchors_per_level[i]) # Number of anchors preset for each grid in this level
            target = torch.zeros(
                self.num_std_channels, S, num_anchors_in_level, 3 + self.num_classes
            )
            target_tensors.append(target)
        
        # 4. Iterate over annotation boxes and fill target tensor
        for bbox in bboxes:
            wave_type = bbox.get("wave_type")
            if wave_type not in self.classes: continue

            original_channel_indices = bbox["channel_idx"] # Corresponds to standard 10-20 arrangement position

            class_idx = self.class2idx[wave_type]
            time_range = bbox["time_range"]

            start_point = time_range[0] * self.sampling_rate
            end_point = time_range[1] * self.sampling_rate
            
            center_x = (start_point + end_point) / 2
            width_w = end_point - start_point
            
            if width_w <= 0: continue

            center_x_norm = center_x / self.seq_len
            width_w_norm = width_w / self.seq_len

            # Find the best anchor size
            best_iou, best_level_idx, best_anchor_idx_in_level = -1, -1, -1
            for level_idx, anchors_in_level in enumerate(self.anchors_per_level):
                for anchor_idx, anchor_w in enumerate(anchors_in_level):
                    intersection = min(width_w_norm, anchor_w)
                    union = width_w_norm + anchor_w - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_level_idx, best_anchor_idx_in_level = iou, level_idx, anchor_idx

            S = self.S_LEVELS[best_level_idx]
            target_anchor_w = self.anchors_per_level[best_level_idx][best_anchor_idx_in_level]
            
            grid_idx = int(S * center_x_norm)
            tx_target = S * center_x_norm - grid_idx
            tw_target = math.log(width_w_norm / target_anchor_w + 1e-6)

            target_to_fill = target_tensors[best_level_idx]
            for original_ch_idx in original_channel_indices:
                # Use mapping to find the new physical row of the channel after shuffling
                current_physical_row = original_to_current_map[original_ch_idx].item()
                if target_to_fill[current_physical_row, grid_idx, best_anchor_idx_in_level, 2] == 0:
                    target_to_fill[current_physical_row, grid_idx, best_anchor_idx_in_level, 0] = tx_target
                    target_to_fill[current_physical_row, grid_idx, best_anchor_idx_in_level, 1] = tw_target
                    target_to_fill[current_physical_row, grid_idx, best_anchor_idx_in_level, 2] = 1.0
                    target_to_fill[current_physical_row, grid_idx, best_anchor_idx_in_level, 3 + class_idx] = 1.0

        return eeg_tensor, target_tensors, attention_mask, pos_indices