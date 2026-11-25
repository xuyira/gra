from collections import defaultdict
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.ndimage import label, binary_closing
from scipy.stats import kurtosis
import sys
from pathlib import Path
import json
from scipy.signal import find_peaks

CH_NAMES_DEFAULT = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

def _mad(x, axis=1):
    return np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)

def _mat_iqr(x, axis=None):
    if axis is None:
        q1, q3 = np.percentile(x, [25, 75])
    else:
        q1, q3 = np.percentile(x, [25, 75], axis=axis)
    return q3 - q1

IQR_TO_SD, MAD_TO_SD = 0.7413, 1.4826
def calculate_iou(range1: list[float], range2: list[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two time intervals.
    
    Args:
        range1/2 (list[float]): Time interval, e.g. [0.2, 1.5]
    """
    start1, end1 = range1
    start2, end2 = range2
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_duration = max(0, intersection_end - intersection_start)
    if intersection_duration == 0:
        return 0.0
    union_duration = (end1 - start1) + (end2 - start2) - intersection_duration
    if union_duration == 0:
        return 0.0
    return intersection_duration / union_duration
class EEGArtifact:
    def __init__(self, data:np.ndarray, freq:int, ch_order:List[str], verbose:bool=False, **kwargs):
        self.data, self.freq, self.ch_order, self.verbose = data, freq, ch_order, verbose
        self.params = {
            "T_artifact_veto_overlap_ratio": 0.8,
            "T_severe_artifact_min_ptp_uv": 80.0,
            "T_severe_artifact_abs_amp_thresh_uv": 200.0,
            "T_eog_min_duration_ms": 94,
            "T_eog_max_duration_ms": 3000,
            "T_eog_biphasic_skip_duration_ms": 370,
            "T_eog_biphasic_preceding_neg_ratio": 0.58,  
            "T_eog_biphasic_succeeding_neg_ratio": 0.58,
            "debug_muscle_channel_of_interest": None,
            "T_slid_corr_symmetric_corr_thresh": 0.63, 
            "T_severe_artifact_min_duration_s": 0.7,
            "T_slid_corr_enable": True,                                 
            "T_slid_corr_window_s": 1.0,                                
            "T_slid_corr_step_s": 0.2,                                 
            "T_slid_corr_lpf_cutoff": 50.0,                            
            "T_slid_corr_threshold": 0.12,                              
            "T_slid_corr_min_duration_s": 0.5,
            "T_muscle_rhythmicity_check_enabled": True,
            "T_muscle_rhythmicity_freq_band_hz": [4, 20],   
            "T_muscle_spectral_concentration_thresh": 8,
            "T_muscle_use_explicit_merging": True,   
            "T_muscle_merge_max_gap_s": 0.8, 
            "T_pre_nan_inf_eog_mask_s": 1.5,
            "T_post_nan_inf_eog_mask_s": 1.5,
            "T_muscle_slow_wave_suppression_enabled": True,    
            "T_muscle_slow_wave_max_freq_hz": 4,          
            "T_muscle_hf_to_slow_wave_amp_ratio": 0.00,
            "T_eog_raw_ch_abs_peak_thresh_v": 40.0, 
            "T_eog_raw_ch_abs_peak_thresh_h": 50.0,
            "T_muscle_max_kurtosis": 100.0,
            "T_muscle_min_volatility_uV": 0.0,
            "T_veog_min_duration_ms": 94,      
            "T_veog_max_duration_ms": 3000,    
            "T_heog_min_duration_ms": 200,      
            "T_heog_max_duration_ms": 3000,
            "T_eog_max_duration_ms": 3000,
            "T_eog_merge_max_gap_s": 0.01,
            "T_eog_abs_peak_thresh_v": 40,  
            "T_eog_abs_peak_thresh_h": 50,
            "EOG_use_relative_amp_thresh": True,    
            "EOG_relative_amp_multiplier": 4,     
            "T_resp_min_amp_uV": 20.0,
            "respiration_half_period_range_s": [0.9, 2.0], 
            "respiration_period_stability_thresh_s": 0.5, 
            "respiration_min_alternating_peaks": 6, 
            "window_size": 0.7, "step_size": 0.1, "T_flat": 2.2, "T_range": [-300, 300],
            "T_roughness_zsd": 55, "T_ampzsd": 10, "T_muscle_power_ratio": 0.28,"T_muscle_min_uV_rms": 3, 
            "T_global_bad_ch_std_uV": [0.5, 250.0], "T_global_bad_ch_corr": 0.1,
            "max_global_bad_ch_fraction": 0.3,
            "artifact_closing_iterations": {
                "default": 1,           
                "eog_v": 0,
                "eog_left": 0,
                "eog_right": 0,
                "drowsiness": 0,
                "muscle": 3,
                "respiration": 2,
                "amp": 2,
            },
            "T_eog_p2p_amp_v": 60,
            "T_eog_p2p_amp_h": 60, 
            "T_eog_p2p_amp_v_max": 350.0, 
            "T_eog_p2p_amp_h_max": 350.0,
            "T_eog_v_corr_thresh": 0.5,  
            "T_eog_h_corr_thresh": -0.5,
            "T_eog_peak_centrality": 0.15, 
            "T_flat_min_duration_s": 2.5,
            "bad_ch_corr_lpf_cutoff": 40.0,
            "T_muscle_min_duration_s": 0.0, 
            "T_muscle_final_min_duration_s": 0.2,
        }
        self.params.update(kwargs)
        self._global_eog_candidates = {}

        self.window_size_s, self.step_size_s = self.params['window_size'], self.params['step_size']
        self.window_size = int(self.window_size_s * self.freq)
        self.step_size = int(self.step_size_s * self.freq)
        self.num_channels, self.num_samples = self.data.shape
        if self.num_samples < self.window_size:
            self.start_indices = np.array([])
        else:
            start_indices = np.arange(0, self.num_samples - self.window_size + 1, self.step_size)
            final_start_index = self.num_samples - self.window_size
            if len(start_indices) == 0 or final_start_index > start_indices[-1]:
                start_indices = np.append(start_indices, final_start_index)
            self.start_indices = start_indices
        
        self.window_num = len(self.start_indices)

        self.global_bad_channels_indices = []
        self.masks = {}
        self.precise_eog_events = []
        self.respiration_mask_timeseries = np.zeros(self.num_samples, dtype=bool)

        try:
            self.fp1_idx, self.fp2_idx = self.ch_order.index('Fp1'), self.ch_order.index('Fp2')
            self.f7_idx, self.f8_idx = self.ch_order.index('F7'), self.ch_order.index('F8')
        except ValueError: 
            self.fp1_idx, self.fp2_idx, self.f7_idx, self.f8_idx = None, None, None, None

        if self.window_num > 0:
            self.run_detection()
        else:
            if self.num_samples > 0:
                print("Warning: Data is too short for windowing. Skipping detection.")

    from scipy.ndimage import label

    def _merge_bboxes(self, bboxes: List[Dict], max_gap_s: float) -> List[Dict]:
        if not bboxes:
            return []

        from collections import defaultdict
        grouped_bboxes = defaultdict(list)
        for bbox in bboxes:
            key = (bbox['channel_idx'], bbox['wave_type'])
            grouped_bboxes[key].append(bbox)

        final_merged_bboxes = []
        for key, group in grouped_bboxes.items():
            wave_type = key[1]
            if (wave_type != 'severe_artifact' and "eog" not in wave_type) or len(group) < 2:
                final_merged_bboxes.extend(group)
                continue
            sorted_group = sorted(group, key=lambda x: x['time_range'][0])
            
            merged_list = [sorted_group[0]]
            for current_box in sorted_group[1:]:
                last_merged_box = merged_list[-1]
                
                gap = current_box['time_range'][0] - last_merged_box['time_range'][1]
                if gap < max_gap_s:
                    last_merged_box['time_range'][1] = max(last_merged_box['time_range'][1], current_box['time_range'][1])
                else:
                    merged_list.append(current_box)
            
            final_merged_bboxes.extend(merged_list)
            
        return final_merged_bboxes
    def _detect_sliding_correlation_artifacts(self) -> np.ndarray:
        """
        Detect low correlation with neighboring channels using a sliding window, and use symmetry checks to exclude physiological activity, identifying transient local artifacts.
        Returns:
            A boolean artifact mask of shape (n_channels, n_samples).
        """
        if not self.params.get("T_slid_corr_enable", False):
            return np.zeros((self.num_channels, self.num_samples), dtype=bool)
        window_s = self.params['T_slid_corr_window_s']
        step_s = self.params['T_slid_corr_step_s']
        lpf_cutoff = self.params['T_slid_corr_lpf_cutoff']
        corr_thresh = self.params['T_slid_corr_threshold']
        min_duration_s = self.params['T_slid_corr_min_duration_s']
        symmetric_corr_thresh = self.params.get('T_slid_corr_symmetric_corr_thresh', 0.65)
        
        if self.verbose:
            print("\n--- [Detection Module] Starting sliding window correlation detection (with symmetry check) ---")
            print(f"  - Window size: {window_s}s, Step: {step_s}s, Neighbor correlation threshold: < {corr_thresh}")
            print(f"  - Minimum duration: > {min_duration_s}s, Symmetry exemption threshold: > {symmetric_corr_thresh}")
        SYMMETRIC_MAP = {
            'Fp1': 'Fp2', 'Fp2': 'Fp1', 'F3': 'F4', 'F4': 'F3', 'C3': 'C4', 'C4': 'C3',
            'P3': 'P4', 'P4': 'P3', 'O1': 'O2', 'O2': 'O1', 'F7': 'F8', 'F8': 'F7',
            'T3': 'T4', 'T4': 'T3', 'T5': 'T6', 'T6': 'T5'
        }
        filtered_data = self._get_low_pass_filtered_data(cutoff_freq=lpf_cutoff)
        win_samples = int(window_s * self.freq)
        step_samples = int(step_s * self.freq)
        sliding_starts = np.arange(0, self.num_samples - win_samples + 1, step_samples)
        num_windows = len(sliding_starts)
        correlation_matrix = np.ones((self.num_channels, num_windows))
        NEIGHBORS = {'Fp1':['Fp2','F7','F3'],'Fp2':['Fp1','F8','F4'],'F7':['Fp1','T3','F3'],'F3':['Fp1','F7','C3','Fz'],'Fz':['FP1','FP2'],'C3':['F3','T3','P3','Cz'],'P3':['C3','T5','O1','Pz'],'O1':['P3','O2','T5'],'O2':['P4','O1','T6'],'F4':['Fp2','Fz','C4','F8'],'C4':['F4','T4','P4','Cz'],'P4':['C4','T6','O2','Pz'],'F8':['Fp2','F4','T4'],'T3':['F7','C3','T5'],'T4':['F8','C4','T6'],'T5':['T3','P3','O1'],'T6':['T4','P4','O2'],'Cz':['C3','C4'],'Pz':['P3','P4']}
        good_global_ch_indices = [i for i in range(self.num_channels) if i not in self.global_bad_channels_indices]
        if self.verbose: print("\n  --- [Stage 1: Calculate correlation per window and perform symmetry check] ---")
        for i_win, start_spl in enumerate(sliding_starts):
            end_spl = start_spl + win_samples
            for i_ch in range(self.num_channels):
                if i_ch in self.global_bad_channels_indices: continue
                ch_name = self.ch_order[i_ch]
                if ch_name not in NEIGHBORS: continue
                neighbor_indices = [self.ch_order.index(n) for n in NEIGHBORS[ch_name] if n in self.ch_order and self.ch_order.index(n) in good_global_ch_indices]
                if not neighbor_indices: continue
                ch_data_win = filtered_data[i_ch, start_spl:end_spl]
                corrs = [np.corrcoef(ch_data_win, filtered_data[n_idx, start_spl:end_spl])[0, 1] for n_idx in neighbor_indices]
                min_abs_corr = np.min(np.abs(np.nan_to_num(corrs)))
                is_low_corr_with_neighbors = min_abs_corr < corr_thresh
                symmetric_corr_val = None
                if is_low_corr_with_neighbors:
                    symmetric_ch_name = SYMMETRIC_MAP.get(ch_name)
                    if symmetric_ch_name and symmetric_ch_name in self.ch_order:
                        symmetric_ch_idx = self.ch_order.index(symmetric_ch_name)
                        if symmetric_ch_idx in good_global_ch_indices:
                            symmetric_corr_val = np.corrcoef(ch_data_win, filtered_data[symmetric_ch_idx, start_spl:end_spl])[0, 1]
                            if np.abs(symmetric_corr_val) > symmetric_corr_thresh:
                                min_abs_corr = 1.0
                
                correlation_matrix[i_ch, i_win] = min_abs_corr
                if self.verbose:
                    is_bad_final = min_abs_corr < corr_thresh
                    decision_str = "❌ Low correlation" if is_bad_final else "✅ Normal"
                    sym_corr_str = f"|SymCorr={np.abs(symmetric_corr_val):.2f}|" if symmetric_corr_val is not None else ""
                    print(f"  [Debug] Win#{i_win:<3} | Channel:{ch_name:<5} | Min|N-corr|={min_abs_corr:.3f} {sym_corr_str} -> {decision_str}")
                
        bad_windows_mask = correlation_matrix < corr_thresh
        min_len_windows = max(1, int(min_duration_s / step_s))
        
        persistent_bad_mask = np.zeros_like(bad_windows_mask, dtype=bool)
        if self.verbose: print("\n  --- [Stage 2: Duration filtering] ---")
        for i_ch in range(self.num_channels):
            labeled_array, num_features = label(bad_windows_mask[i_ch, :])
            if self.verbose and num_features > 0:
                ch_name = self.ch_order[i_ch]
                print(f"  [Debug] Channel: {ch_name:<5}")
                print(f"    - Initially found {num_features} independent low-correlation blocks.")
                component_sizes = np.bincount(labeled_array.ravel())[1:]
                print(f"    - Block lengths (windows): {component_sizes}")
                print(f"    - Minimum required length (windows): >={min_len_windows}")
                long_enough_labels = np.where(component_sizes >= min_len_windows)[0] + 1
                if len(long_enough_labels) > 0: print(f"    -> Decision: Keep blocks {long_enough_labels}.")
                else: print("    -> Decision: No blocks long enough, discard all.")
            if num_features > 0:
                component_sizes = np.bincount(labeled_array.ravel())[1:]
                long_enough_labels = np.where(component_sizes >= min_len_windows)[0] + 1
                if len(long_enough_labels) > 0:
                    persistent_bad_mask[i_ch, :] = np.isin(labeled_array, long_enough_labels)

        final_artifact_mask = np.zeros((self.num_channels, self.num_samples), dtype=bool)
        for i_ch in range(self.num_channels):
            for i_win in np.where(persistent_bad_mask[i_ch, :])[0]:
                start_spl = sliding_starts[i_win]
                end_spl = start_spl + win_samples
                final_artifact_mask[i_ch, start_spl:end_spl] = True

        if self.verbose:
            num_artifact_events = np.sum([label(ch_mask)[1] for ch_mask in final_artifact_mask])
            print(f"\n  -> Sliding correlation detection finished. Found {num_artifact_events} final low-correlation artifact events.")

        return final_artifact_mask

    def _apply_nan_inf_boundary_suppression(self):
        pre_mask_s = self.params.get("T_pre_nan_inf_eog_mask_s", 0.5)
        post_mask_s = self.params.get("T_post_nan_inf_eog_mask_s", 0.5)

        if pre_mask_s <= 0 and post_mask_s <= 0:
            return 

        pre_mask_windows = int(np.ceil(pre_mask_s / self.step_size_s))
        post_mask_windows = int(np.ceil(post_mask_s / self.step_size_s))
        if self.verbose:
            print(f"  -> Applying nan/inf boundary EOG suppression (pre: {pre_mask_s}s, post: {post_mask_s}s).")

        for i_ch in range(self.num_channels):
            ch_nan_inf_mask = self.masks['nan_inf'][i_ch, :]
            if not np.any(ch_nan_inf_mask):
                continue 

            diff_mask = np.diff(np.concatenate(([0], ch_nan_inf_mask.astype(int), [0])))
            onset_windows = np.where(diff_mask == 1)[0]
            offset_windows = np.where(diff_mask == -1)[0]

            if pre_mask_windows > 0:
                for start_artifact in onset_windows:
                    start_suppress = max(0, start_artifact - pre_mask_windows)
                    end_suppress = start_artifact
                    self.masks['eog_v_positive'][i_ch, start_suppress:end_suppress] = False
                    self.masks['eog_left'][i_ch, start_suppress:end_suppress] = False
                    self.masks['eog_right'][i_ch, start_suppress:end_suppress] = False

            if post_mask_windows > 0:
                for start_suppress in offset_windows:
                    end_suppress = start_suppress + post_mask_windows
                    self.masks['eog_v_positive'][i_ch, start_suppress:end_suppress] = False
                    self.masks['eog_left'][i_ch, start_suppress:end_suppress] = False
                    self.masks['eog_right'][i_ch, start_suppress:end_suppress] = False

    def get_all_masks_v2(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, 'masks') or not self.masks:
            return {}
        processed_masks = {}
        closing_config = self.params.get("artifact_closing_iterations", {})
        default_iters = closing_config.get("default", 1)
        
        use_explicit_merging = self.params.get("T_muscle_use_explicit_merging", True)

        for mask_label, mask in self.masks.items():
            if mask_label == 'muscle' and use_explicit_merging:
                processed_masks[mask_label] = mask
                continue
            iterations = closing_config.get(mask_label, default_iters)
            if iterations > 0:
                if self.verbose: print(f"-> Merging '{mask_label}' artifacts with morphology closing ({iterations} iterations)...")
                processed_masks[mask_label] = self._postprocess_mask(mask, iterations=iterations)
            else:
                processed_masks[mask_label] = mask
        
        if use_explicit_merging and 'muscle' in processed_masks:
            max_gap_s = self.params.get("T_muscle_merge_max_gap_s", 0.5)
            if self.verbose: print(f"-> Merging 'muscle' artifacts with explicit gap threshold (visual gap < {max_gap_s}s)...")

            compensated_gap_s = max_gap_s + self.window_size_s - self.step_size_s
            max_gap_windows = compensated_gap_s / self.step_size_s

            muscle_mask = processed_masks['muscle']
            merged_muscle_mask = np.zeros_like(muscle_mask)

            for i_ch in range(self.num_channels):
                ch_mask = muscle_mask[i_ch, :]
                if not np.any(ch_mask):
                    continue

                diff_mask = np.diff(np.concatenate(([0], ch_mask.astype(int), [0])))
                onset_wins = np.where(diff_mask == 1)[0]
                offset_wins = np.where(diff_mask == -1)[0]

                if len(onset_wins) <= 1:
                    merged_muscle_mask[i_ch, :] = ch_mask
                    continue

                current_start = onset_wins[0]
                current_end = offset_wins[0]

                for i in range(1, len(onset_wins)):
                    next_start = onset_wins[i]
                    gap = next_start - current_end
                    
                    if self.verbose:
                        gap_s = gap * self.step_size_s
                        visual_gap_s = gap_s - (self.window_size_s - self.step_size_s)
                        print(f"  [MUSCLE MERGE DEBUG] Ch:{i_ch} | "
                            f"Mask gap is {gap} windows ({gap_s:.3f}s), which translates to a VISUAL gap of ~{visual_gap_s:.3f}s. "
                            f"Threshold is visual_gap <= {max_gap_s:.3f}s.")

                    if gap <= max_gap_windows:
                        if self.verbose: print("    -> Decision: MERGING.")
                        current_end = offset_wins[i]
                    else:
                        if self.verbose: print("    -> Decision: NOT MERGING.")
                        merged_muscle_mask[i_ch, current_start:current_end] = True
                        current_start = next_start
                        current_end = offset_wins[i]
                
                merged_muscle_mask[i_ch, current_start:current_end] = True
            
            processed_masks['muscle'] = merged_muscle_mask
        flat_min_dur_s = self.params.get('T_flat_min_duration_s', 3.0)
        if flat_min_dur_s > 0 and 'flat' in processed_masks:
            min_len_windows = flat_min_dur_s / self.step_size_s
            raw_flat_mask = processed_masks['flat']
            duration_filtered_flat_mask = np.zeros_like(raw_flat_mask)
            for i_ch in range(self.num_channels):
                labeled_array, num_features = label(raw_flat_mask[i_ch, :])
                if num_features == 0: continue
                component_sizes = np.bincount(labeled_array.ravel())
                long_enough_labels = np.where(component_sizes[1:] >= min_len_windows)[0] + 1
                if len(long_enough_labels) > 0:
                    ch_final_flat_mask = np.isin(labeled_array, long_enough_labels)
                    duration_filtered_flat_mask[i_ch, :] = ch_final_flat_mask
            processed_masks['flat'] = duration_filtered_flat_mask
        priority_groups = [["eog_left", "eog_right"],["eog_v"],["severe_artifact"],["nan_inf"],["muscle"],[ "flat"],
    [ "drowsiness","global_bad" ]]
        final_masks = {}
        already_marked_by_higher_groups = np.zeros((self.num_channels, self.window_num), dtype=bool)
        for group in priority_groups:
            marked_within_this_group = np.zeros_like(already_marked_by_higher_groups)
            for mask_label in group:
                if mask_label not in processed_masks:
                    continue
                raw_mask = processed_masks[mask_label]
                current_mask = raw_mask & ~already_marked_by_higher_groups
                final_masks[mask_label] = current_mask
                marked_within_this_group |= current_mask
            already_marked_by_higher_groups |= marked_within_this_group
                
        return final_masks

    def _detect_respiration_morphological(self) -> np.ndarray:

        # --- 1. Initialization and parameter loading ---
        WINDOW_S = 2.5       
        SLACK = 0.15        
        MIN_INTERVAL_S = 0.8 
        min_n_peaks = self.params.get('respiration_min_alternating_peaks', 4)
        period_range_s = self.params.get('respiration_half_period_range_s', [0.9, 2.0])
        stability_thresh_s = self.params.get('respiration_period_stability_thresh_s', 0.3)
        T_resp_min_amp_uV = self.params.get("T_resp_min_amp_uV", 20.0)
        if self.verbose:
            print("\n" + "="*20 + " [Debug] Respiration artifact detection " + "="*20)
            print(f"  - Parameters: min alternating peaks={min_n_peaks}, half-period range={period_range_s}s, period stability threshold={stability_thresh_s}s, min amplitude={T_resp_min_amp_uV}uV")

        good_indices = [i for i in range(self.num_channels) if i not in self.global_bad_channels_indices]
        if len(good_indices) < 2: 
            if self.verbose: print("  - Decision: Not enough available channels (<2), skipping detection.")
            return np.zeros(self.num_samples, dtype=bool)

    # --- 2. Signal preprocessing ---
        mean_signal = np.mean(self.data[good_indices, :], axis=0)
        nyquist = 0.5 * self.freq
        lowcut_hz = 1.0
        if nyquist > lowcut_hz:
            b, a = butter(4, lowcut_hz / nyquist, btype='low')
            signal_for_detection = filtfilt(b, a, mean_signal)
            if self.verbose: print(f"  - Signal preprocessing: Applied {lowcut_hz}Hz low-pass filter.")
        else:
            signal_for_detection = mean_signal
            if self.verbose: print("  - Signal preprocessing: No filtering applied.")

    # --- 3. Iteratively find peak sequence ---
        sequence = []
        current_pos = 0
        target_polarity = None
        window_samples = int(WINDOW_S * self.freq)
        min_interval_samples = int(MIN_INTERVAL_S * self.freq)
        iteration_count = 0

        if self.verbose: print("\n--- [Stage 1] Start iterative search for peak sequence ---")
        
        while current_pos < self.num_samples:
            iteration_count += 1
            if self.verbose: print(f"\n--- [Iteration #{iteration_count}] ---")

            start = current_pos
            end = min(current_pos + window_samples, self.num_samples)
            
            if (end - start) < self.freq:
                if self.verbose: print(f"  - Window: [{start/self.freq:.2f}s - {end/self.freq:.2f}s] -> ❌ Failed: Remaining data too short, stop searching.")
                break
            
            if self.verbose: print(f"  - Window: [{start/self.freq:.2f}s - {end/self.freq:.2f}s]")
            window_signal = signal_for_detection[start:end]
            win_max_val, win_min_val = np.max(window_signal), np.min(window_signal)
            target_amp = max(abs(win_max_val), abs(win_min_val))
            threshold = target_amp * (1 - SLACK)
            if self.verbose: print(f"  - Window threshold: {threshold:.2f} uV (based on peak {target_amp:.2f} uV)")

            candidate_indices_local, = np.where(np.abs(window_signal) >= threshold)

            if len(candidate_indices_local) == 0:
                if self.verbose: print("  - Decision: ❌ Failed: No significant points in window. Slide window forward.")
                current_pos += window_samples // 2 
                continue

            win_mean = np.mean(window_signal)
            polarity_filtered_candidates = []
            for idx in candidate_indices_local:
                polarity = 1 if window_signal[idx] >= win_mean else -1
                if target_polarity is None or polarity == target_polarity:
                    polarity_filtered_candidates.append(idx)
            
            if self.verbose: print(f"  - Candidate points: found {len(candidate_indices_local)}, after polarity filtering {len(polarity_filtered_candidates)} (target polarity: {target_polarity})")

            if not polarity_filtered_candidates:
                if self.verbose: print("  - Decision: ❌ Failed: Significant points found but polarity does not match. Slide window forward.")
                current_pos += window_samples // 2
                continue
                
            best_local_idx = min(polarity_filtered_candidates)
            best_global_idx = start + best_local_idx
            
            if sequence and (best_global_idx - sequence[-1][0]) < min_interval_samples:
                if self.verbose: print(f"  - Decision: ❌ Failed: Found peak too close to previous one. Continue searching.")
                current_pos = best_global_idx + 1
                continue

            best_val = signal_for_detection[best_global_idx]
            sequence.append((best_global_idx, best_val))
            current_polarity = 1 if best_val >= win_mean else -1
            target_polarity = -current_polarity
            if self.verbose: 
                print(f"  - Decision: ✅ Selected peak at {best_global_idx/self.freq:.2f}s (value: {best_val:.2f} uV)")
                print(f"  - State update: sequence length now {len(sequence)}, next target polarity is {target_polarity}")

            current_pos = best_global_idx + 1
        
        # --- 4. Final validation ---
        if self.verbose: 
            print("\n" + "-"*40)
            print(f"--- [Stage 2] Loop finished: found {len(sequence)} alternating peaks ---")
            print("--- Starting final validation of the whole sequence ---")
        
        if len(sequence) < min_n_peaks:
            if self.verbose: print(f"  - Check [sequence length]: {len(sequence)} (required: >= {min_n_peaks}) -> ❌ Failed")
            return np.zeros(self.num_samples, dtype=bool)
        else:
            if self.verbose: print(f"  - Check [sequence length]: {len(sequence)} (required: >= {min_n_peaks}) -> ✅ Passed")

        indices = np.array([p[0] for p in sequence])
        intervals_sec = np.diff(indices) / self.freq
        mean_interval = np.mean(intervals_sec)
        std_interval = np.std(intervals_sec)
        abs_amplitudes = np.abs([p[1] for p in sequence])
        mean_abs_amp = np.mean(abs_amplitudes)

        is_period_appropriate = period_range_s[0] <= mean_interval <= period_range_s[1]
        is_period_stable = std_interval < stability_thresh_s
        is_amp_sufficient = mean_abs_amp >= T_resp_min_amp_uV
        
        if self.verbose:
            print(f"  - Check [periodicity]: mean half-period={mean_interval:.2f}s (range: {period_range_s}) -> {'✅ Passed' if is_period_appropriate else '❌ Failed'}")
            print(f"  - Check [stability]: period std={std_interval:.2f}s (threshold: < {stability_thresh_s}) -> {'✅ Passed' if is_period_stable else '❌ Failed'}")
            print(f"  - Check [amplitude]: mean peak amplitude={mean_abs_amp:.2f}uV (threshold: >= {T_resp_min_amp_uV}) -> {'✅ Passed' if is_amp_sufficient else '❌ Failed'}")
        
        final_decision = is_period_appropriate and is_period_stable and is_amp_sufficient
        
        # --- 5. Generate result ---
        if self.verbose: print(f"\n--- [Final conclusion]: {'⭐⭐⭐ Respiration artifact detected ⭐⭐⭐' if final_decision else '--- No respiration artifact detected ---'}")
        
        final_resp_mask = np.zeros(self.num_samples, dtype=bool)
        if final_decision:
            final_resp_mask[indices[0]:indices[-1]] = True
            if self.verbose: print(f"  - Artifact interval marked: from {indices[0]/self.freq:.2f}s to {indices[-1]/self.freq:.2f}s")
                
        return final_resp_mask
    def _get_band_pass_filtered_data(self, lowcut, highcut, order=4) -> np.ndarray:
        """Perform bandpass filtering on the data"""
        nyquist = 0.5 * self.freq
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, self.data, axis=1)

    def _identify_sustained_hf_activity(self) -> np.ndarray:

        min_duration_s = self.params.get("T_muscle_min_duration_s", 0.5)
        min_len_samples = int(min_duration_s * self.freq)

        HF_AMP_THRESH_MAD_FACTOR = 4.0

        lowcut = 45.0
        nyquist = 0.5 * self.freq
        if nyquist <= lowcut: return np.zeros((self.num_channels, self.num_samples), dtype=bool)
        
        b, a = butter(4, lowcut / nyquist, btype='high')
        hf_data = filtfilt(b, a, self.data, axis=1)
        mad_per_channel = _mad(hf_data, axis=1)
        thresholds = mad_per_channel[:, np.newaxis] * HF_AMP_THRESH_MAD_FACTOR
        
        activity_mask = np.abs(hf_data) > thresholds
        
        final_mask = np.zeros_like(activity_mask, dtype=bool)
        for i_ch in range(self.num_channels):
            labeled_array, num_features = label(activity_mask[i_ch, :])
            if num_features == 0:
                continue
            component_sizes = np.bincount(labeled_array.ravel())[1:] 
            long_enough_labels = np.where(component_sizes >= min_len_samples)[0] + 1
            
            if len(long_enough_labels) > 0:
                ch_final_mask = np.isin(labeled_array, long_enough_labels)
                final_mask[i_ch, :] = ch_final_mask
                
        return final_mask
    def _is_segment_sharp(self, segment: np.ndarray, slope_thresh_uv_ms: float) -> Tuple[float, bool]:
        """
        Check if the maximum slope of a signal segment exceeds the threshold.
        Returns: (calculated max slope value, boolean indicating if threshold is exceeded)
        """
        if segment.size < 2:
            return 0.0, False
        max_slope = np.max(np.abs(np.diff(segment))) * self.freq / 1000.0
        is_sharp = max_slope > slope_thresh_uv_ms
        return max_slope, is_sharp

    from scipy.ndimage import label
    def _find_eog_candidates_globally(self, signal: np.ndarray, signal_name: str, params: Dict) -> List[Dict]:
        """
        Perform a one-time search for candidate events on the full signal.
        New logic: Segment potential biphasic waves by finding peaks, return event list with stable boundaries.
        """
        min_dur_s = params['T_eog_min_duration_ms'] / 1000.0
        max_dur_s = params['T_eog_max_duration_ms'] / 1000.0
        abs_peak_thresh = params['T_eog_abs_peak_thresh']
        rel_ptp_thresh = params['T_rel_ptp_thresh']
        max_ptp = params['T_max_ptp']

        if self.verbose:
            print(f"\n--- Global candidate search (signal: {signal_name}) ---")
        
        initial_thresh = rel_ptp_thresh / 3 if rel_ptp_thresh > 0 else abs_peak_thresh / 3
        if initial_thresh == 0: initial_thresh = 10
        
        candidate_points = np.abs(signal) > initial_thresh
        labeled_array, num_features = label(candidate_points)
        
        if self.verbose:
            print(f"  - Using the initial threshold {initial_thresh:.2f} uV, {num_features} candidate areas are found.")
        if num_features == 0: return []
        
        all_peak_based_events = []
        for i in range(1, num_features + 1):
            event_indices = np.where(labeled_array == i)[0]
            if len(event_indices) <= 3: continue
            
            core_segment = signal[event_indices[0]:event_indices[-1]+1]
    
            peaks_pos, _ = find_peaks(core_segment, height=initial_thresh, distance=int(0.05 * self.freq))
            peaks_neg, _ = find_peaks(-core_segment, height=initial_thresh, distance=int(0.05 * self.freq))
            
            all_peaks = sorted(np.concatenate([peaks_pos, peaks_neg]))

            for peak_idx_local in all_peaks:
                peak_idx_global = event_indices[0] + peak_idx_local
                start_final, end_final = peak_idx_global, peak_idx_global
                current_sign = np.sign(signal[peak_idx_global])
                if current_sign == 0: continue

                while start_final > 0 and np.sign(signal[start_final - 1]) == current_sign:
                    start_final -= 1
                while end_final < len(signal) - 1 and np.sign(signal[end_final + 1]) == current_sign:
                    end_final += 1
                
                all_peak_based_events.append({'start': start_final, 'end': end_final})

        if not all_peak_based_events: return []
        
        unique_events_set = {tuple(d.items()) for d in all_peak_based_events}
        unique_events = [dict(t) for t in unique_events_set]
        sorted_events = sorted(unique_events, key=lambda x: x['start'])
        
        if self.verbose:
            print(f"  - Identified {len(sorted_events)} independent peak events from the candidate area for preliminary verification...")

        valid_candidates = []
        for event in sorted_events:
            start_idx, end_idx = event['start'], event['end']
            event_segment = signal[start_idx:end_idx+1]
            if len(event_segment) == 0: continue
            
            duration_s = (end_idx - start_idx + 1) / self.freq
            if not (min_dur_s <= duration_s <= max_dur_s): continue
                
            if not (np.max(np.abs(event_segment)) >= abs_peak_thresh): continue
            if not (rel_ptp_thresh <= np.ptp(event_segment) <= max_ptp): continue

            valid_candidates.append(event)

        if self.verbose:
            print(f"  -> Confirmed {len(valid_candidates)} global candidate events.")

        return valid_candidates

    def _detect_eog_and_drowsiness_v2(self):
        """
        Main function for EOG detection. Uses "global search first, then window-by-window validation" strategy.
        This function is called once by run_detection, directly modifies self.masks and self.precise_eog_events.
        """
        if self.fp1_idx is None or self.fp2_idx is None:
            if self.verbose: print("\n[EOG Detection] Skipped due to missing Fp1/Fp2 channel.")
            return

        if self.verbose:
            print("\n" + "="*80)
            print(f"--- EOG Detection Module (Global Search, Window-by-Window Validation Mode) ---")
        biphasic_skip_duration_ms = self.params.get("T_eog_biphasic_skip_duration_ms", 300)
        preceding_neg_ratio = self.params.get("T_eog_biphasic_preceding_neg_ratio", 0.6)
        succeeding_neg_ratio = self.params.get("T_eog_biphasic_succeeding_neg_ratio", 0.6)
        biphasic_check_enabled = self.params.get("T_eog_biphasic_check_enable", True)
        lookbehind_ms = self.params.get("T_eog_biphasic_lookbehind_ms", 250)
        preceding_neg_thresh_uv = self.params.get("T_eog_biphasic_preceding_neg_thresh_uv", 40.0)
        lookahead_ms = self.params.get("T_eog_biphasic_lookahead_ms", 250)
        succeeding_neg_thresh_uv = self.params.get("T_eog_biphasic_succeeding_neg_thresh_uv", 20.0)
        sharpness_thresh = self.params.get("T_eog_biphasic_sharpness_thresh_uv_ms", 4.3)
        v_corr_thresh = self.params.get('T_eog_v_corr_thresh', 0.6)
        h_corr_thresh = self.params.get('T_eog_h_corr_thresh', -0.65)
        max_roughness = self.params.get("T_eog_max_roughness", 100000.0)
        peak_centrality_thresh = self.params.get("T_eog_peak_centrality", 0.01)
        raw_ch_abs_peak_thresh_v = self.params.get("T_eog_raw_ch_abs_peak_thresh_v", 75.0)
        raw_ch_abs_peak_thresh_h = self.params.get("T_eog_raw_ch_abs_peak_thresh_h", 60.0)
        max_slope_thresh = self.params.get("T_eog_v_max_slope_uv_ms", 20.0)
        rel_amp_multiplier = self.params.get("EOG_relative_amp_multiplier", 5.0)
        if not self._global_eog_candidates: 
            if self.verbose:
                print(f"\n--- EOG Detection Module: First run, start global candidate search (entire data segment) ---")
            
            eog_baselines = {}
            if self.params.get("EOG_use_relative_amp_thresh", True):
                eog_baselines['fp1'] = _mad(self.data[self.fp1_idx, :], axis=None)
                eog_baselines['fp2'] = _mad(self.data[self.fp2_idx, :], axis=None)
                if self.f7_idx is not None and self.f8_idx is not None:
                    eog_baselines['f7'] = _mad(self.data[self.f7_idx, :], axis=None)
                    eog_baselines['f8'] = _mad(self.data[self.f8_idx, :], axis=None)
            rel_ptp_thresh_v = 0
            if eog_baselines.get('fp1', 0) > 0 and eog_baselines.get('fp2', 0) > 0:
                rel_ptp_thresh_v = ((eog_baselines['fp1'] + eog_baselines['fp2']) / 2) * rel_amp_multiplier
            veog_params = {
                            'T_eog_min_duration_ms': self.params.get('T_veog_min_duration_ms', self.params['T_eog_min_duration_ms']),
                            'T_eog_max_duration_ms': self.params.get('T_veog_max_duration_ms', self.params['T_eog_max_duration_ms']),
                            'T_eog_abs_peak_thresh': self.params['T_eog_abs_peak_thresh_v'], 
                            'T_rel_ptp_thresh': rel_ptp_thresh_v, 
                            'T_max_ptp': self.params['T_eog_p2p_amp_v_max']
                        }
            veog_signal_full = (self.data[self.fp1_idx, :] + self.data[self.fp2_idx, :]) / 2
            self._global_eog_candidates['veog'] = self._find_eog_candidates_globally(veog_signal_full, "VEOG", veog_params)

            rel_ptp_thresh_h_fp = 0
            if eog_baselines.get('fp1', 0) > 0 and eog_baselines.get('fp2', 0) > 0:
                rel_ptp_thresh_h_fp = ((eog_baselines['fp1'] + eog_baselines['fp2']) / 2) * rel_amp_multiplier
            heog_fp_params = {
                                'T_eog_min_duration_ms': self.params.get('T_heog_min_duration_ms', self.params['T_eog_min_duration_ms']),
                                'T_eog_max_duration_ms': self.params.get('T_heog_max_duration_ms', self.params['T_eog_max_duration_ms']),
                                'T_eog_abs_peak_thresh': self.params['T_eog_abs_peak_thresh_h'], 
                                'T_rel_ptp_thresh': rel_ptp_thresh_h_fp, 
                                'T_max_ptp': self.params['T_eog_p2p_amp_h_max']
                            }
            heog_signal_fp_full = self.data[self.fp1_idx, :] - self.data[self.fp2_idx, :]
            self._global_eog_candidates['heog_fp'] = self._find_eog_candidates_globally(heog_signal_fp_full, "HEOG_Fp", heog_fp_params)

            if self.f7_idx is not None and self.f8_idx is not None:
                rel_ptp_thresh_h_t = 0
                if eog_baselines.get('f7', 0) > 0 and eog_baselines.get('f8', 0) > 0:
                    rel_ptp_thresh_h_t = ((eog_baselines['f7'] + eog_baselines['f8']) / 2) * rel_amp_multiplier
                heog_t_params = {
                                    'T_eog_min_duration_ms': self.params.get('T_heog_min_duration_ms', self.params['T_eog_min_duration_ms']),
                                    'T_eog_max_duration_ms': self.params.get('T_heog_max_duration_ms', self.params['T_eog_max_duration_ms']),
                                    'T_eog_abs_peak_thresh': self.params['T_eog_abs_peak_thresh_h'], 
                                    'T_rel_ptp_thresh': rel_ptp_thresh_h_t, 
                                    'T_max_ptp': self.params['T_eog_p2p_amp_h_max']
                                }
                heog_signal_t_full = self.data[self.f7_idx, :] - self.data[self.f8_idx, :]
                self._global_eog_candidates['heog_t'] = self._find_eog_candidates_globally(heog_signal_t_full, "HEOG_T", heog_t_params)

        veog_signal_full = (self.data[self.fp1_idx, :] + self.data[self.fp2_idx, :]) / 2
        heog_signal_fp_full = self.data[self.fp1_idx, :] - self.data[self.fp2_idx, :]
        if self.f7_idx is not None and self.f8_idx is not None:
            heog_signal_t_full = self.data[self.f7_idx, :] - self.data[self.f8_idx, :]

        for i_win, start_sample in enumerate(self.start_indices):
            end_sample = start_sample + self.window_size
            if self.verbose: print(f"\n--- Window #{i_win}: Start verifying global candidates overlapping with this window ---")
            for i, candidate in enumerate(self._global_eog_candidates.get('veog', [])):
                cand_start, cand_end = candidate['start'], candidate['end']
                if max(start_sample, cand_start) >= min(end_sample, cand_end): continue
                
                if self.verbose:
                    cand_start_s, cand_end_s = cand_start / self.freq, cand_end / self.freq
                    print(f"\n    --- VEOG Candidate #{i+1} (Absolute time: {cand_start_s:.3f}s-{cand_end_s:.3f}s) ---")
                
                seg1_full, seg2_full = self.data[self.fp1_idx, cand_start:cand_end+1], self.data[self.fp2_idx, cand_start:cand_end+1]
                event_segment_veog_full = veog_signal_full[cand_start:cand_end+1]
                
                corr = np.corrcoef(seg1_full, seg2_full)[0, 1] if len(seg1_full) > 1 else 0.0
                corr_check = corr > v_corr_thresh
                if self.verbose: print(f"      - Check [Correlation Fp1-Fp2]: {corr:.2f} (Threshold: >{v_corr_thresh:.2f}) -> {'✅' if corr_check else '❌'}")
                if not corr_check: continue

                polarity_check = np.max(event_segment_veog_full) > -np.min(event_segment_veog_full)
                if self.verbose: print(f"      - Check [Positive wave dominance]: {np.max(event_segment_veog_full):.1f}uV > -({np.min(event_segment_veog_full):.1f}uV) -> {'✅' if polarity_check else '❌'}")
                if not polarity_check: continue

                avg_roughness = (self.roughnesses[self.fp1_idx, i_win] + self.roughnesses[self.fp2_idx, i_win]) / 2
                roughness_check = avg_roughness <= max_roughness
                if self.verbose: print(f"      - Check [Roughness at win#{i_win}]: {avg_roughness:.1f} (Threshold: <={max_roughness:.1f}) -> {'✅' if roughness_check else '❌'}")
                if not roughness_check: continue

                segment_len = len(event_segment_veog_full)
                peak_idx = np.argmax(event_segment_veog_full)
                centrality_check = (segment_len > 0 and (segment_len * peak_centrality_thresh) < peak_idx < (segment_len * (1 - peak_centrality_thresh)))
                if self.verbose: print(f"      - Check [Peak centrality]: Peak position {peak_idx/segment_len if segment_len>0 else 0:.2f} (Range: {peak_centrality_thresh:.2f}-{1-peak_centrality_thresh:.2f}) -> {'✅' if centrality_check else '❌'}")
                if not centrality_check: continue

                fp1_peak, fp2_peak = (np.max(np.abs(seg1_full)) if len(seg1_full)>0 else 0), (np.max(np.abs(seg2_full)) if len(seg2_full)>0 else 0)
                raw_amp_check = fp1_peak >= raw_ch_abs_peak_thresh_v and fp2_peak >= raw_ch_abs_peak_thresh_v
                if self.verbose: print(f"      - Check [Raw channel peak]: Fp1={fp1_peak:.1f}uV, Fp2={fp2_peak:.1f}uV (Threshold: >={raw_ch_abs_peak_thresh_v:.1f}uV) -> {'✅' if raw_amp_check else '❌'}")
                if not raw_amp_check: continue
                
                max_slope = np.max(np.abs(np.diff(event_segment_veog_full))) * self.freq / 1000.0 if len(event_segment_veog_full)>1 else 0
                slope_check = max_slope <= max_slope_thresh
                if self.verbose: print(f"      - Check [Max slope]: {max_slope:.2f} uV/ms (Threshold: <={max_slope_thresh:.1f} uV/ms) -> {'✅' if slope_check else '❌'}")
                if not slope_check: continue

                is_biphasic = False
                if biphasic_check_enabled:
                    positive_peak_amp = np.max(event_segment_veog_full)
                    if self.verbose: print(f"      - Check [Biphasic - preceding]: (Look back {lookbehind_ms}ms)")
                    lookbehind_samples = int(lookbehind_ms * self.freq / 1000.0)
                    lookbehind_start = max(0, cand_start - lookbehind_samples)
                    preceding_rejected = False
                    if lookbehind_start < cand_start:
                        preceding_segment = veog_signal_full[lookbehind_start:cand_start]
                        preceding_neg_amp = -np.min(preceding_segment)
                        dynamic_thresh_pre = positive_peak_amp * preceding_neg_ratio
                        abs_check = preceding_neg_amp >= preceding_neg_thresh_uv
                        rel_check = preceding_neg_amp >= dynamic_thresh_pre
                        if self.verbose:
                            print(f"        - Amplitude (absolute): {preceding_neg_amp:.1f}uV (Threshold: >={preceding_neg_thresh_uv:.1f}uV) -> {'✅' if abs_check else '❌'}")
                            print(f"        - Amplitude (relative): {preceding_neg_amp:.1f}uV (Threshold: >={dynamic_thresh_pre:.1f}uV [{preceding_neg_ratio:.0%}]) -> {'✅' if rel_check else '❌'}")
                        if abs_check and rel_check:
                            slope_val, is_sharp = self._is_segment_sharp(preceding_segment, sharpness_thresh)
                            if self.verbose: print(f"        - Sharpness check: {slope_val:.2f} uV/ms (Threshold: >{sharpness_thresh:.1f} uV/ms) -> {'✅' if is_sharp else '❌'}")
                            if is_sharp: preceding_rejected = True
                    if self.verbose:
                        if preceding_rejected: print("        -> Conclusion: Preceding spike detected, rejected. -> ❌")
                        else: print("        -> Conclusion: No preceding spike detected. -> ✅")
                    if preceding_rejected: is_biphasic = True

                    if not is_biphasic:
                        if self.verbose: print(f"      - Check [Biphasic - succeeding]: (Look ahead {lookahead_ms}ms)")
                        lookahead_samples = int(lookahead_ms * self.freq / 1000.0)
                        lookahead_start = cand_end + 1
                        lookahead_end = min(len(veog_signal_full), lookahead_start + lookahead_samples)
                        succeeding_rejected = False
                        if lookahead_start < lookahead_end:
                            succeeding_segment = veog_signal_full[lookahead_start:lookahead_end]
                            succeeding_neg_amp = -np.min(succeeding_segment)
                            dynamic_thresh_suc = positive_peak_amp * succeeding_neg_ratio
                            abs_check = succeeding_neg_amp >= succeeding_neg_thresh_uv
                            rel_check = succeeding_neg_amp >= dynamic_thresh_suc
                            if self.verbose:
                                print(f"        - Amplitude (absolute): {succeeding_neg_amp:.1f}uV (Threshold: >={succeeding_neg_thresh_uv:.1f}uV) -> {'✅' if abs_check else '❌'}")
                                print(f"        - Amplitude (relative): {succeeding_neg_amp:.1f}uV (Threshold: >={dynamic_thresh_suc:.1f}uV [{succeeding_neg_ratio:.0%}]) -> {'✅' if rel_check else '❌'}")
                                if self.verbose: print(f"        - Sharpness check: {slope_val:.2f} uV/ms (Threshold: >{sharpness_thresh:.1f} uV/ms) -> {'✅' if is_sharp else '❌'}")
                            if abs_check and rel_check:
                                slope_val, is_sharp = self._is_segment_sharp(succeeding_segment, sharpness_thresh)
                                if self.verbose: print(f"        - Sharpness check: {slope_val:.2f} uV/ms (Threshold: >{sharpness_thresh:.1f} uV/ms) -> {'✅' if is_sharp else '❌'}")
                                if is_sharp: succeeding_rejected = True
                        if self.verbose:
                            if succeeding_rejected: print("        -> Conclusion: Succeeding spike detected, rejected. -> ❌")
                            else: print("        -> Conclusion: No succeeding spike detected. -> ✅")
                        if succeeding_rejected: is_biphasic = True
                
                overlapping_wins = [idx for idx, s_start in enumerate(self.start_indices) if max(s_start, cand_start) < min(s_start + self.window_size, cand_end)]

                if is_biphasic:
                    if overlapping_wins:
                        ch_indices_to_mark = [self.fp1_idx, self.fp2_idx]
                        for ch_idx in ch_indices_to_mark:
                            self.masks['eog_v_veto'][ch_idx, overlapping_wins] = True
                
                else:
                    if i_win in overlapping_wins:
                        if self.verbose: print("      - Result: ⭐⭐⭐ VEOG event detected! ⭐⭐⭐")
                        ch_indices_to_mark = [self.fp1_idx, self.fp2_idx]
                        for ch_idx in ch_indices_to_mark:
                            self.masks['eog_v_positive'][ch_idx, i_win] = True
                        self.precise_eog_events.append({
                            'label': 'eog_v', 
                            'ch_indices': ch_indices_to_mark, 
                            'start_sample': cand_start, 
                            'end_sample': cand_end,
                            'window_index': i_win 
                        })

            # Validate HEOG_Fp events
            if 'heog_fp' in self._global_eog_candidates:
                if self.verbose: print("\n" + "#"*30 + " HEOG (Horizontal Eye Movement @ Fp1/Fp2) Validation " + "#"*30)
                for i, candidate in enumerate(self._global_eog_candidates['heog_fp']):
                    cand_start, cand_end = candidate['start'], candidate['end']
                    if max(start_sample, cand_start) >= min(end_sample, cand_end): continue
                    if self.verbose:
                        cand_start_s, cand_end_s = cand_start / self.freq, cand_end / self.freq
                        print(f"\n    --- HEOG_Fp Candidate #{i+1} (Absolute time: {cand_start_s:.3f}s-{cand_end_s:.3f}s) ---")
                    
                    seg1_full, seg2_full = self.data[self.fp1_idx, cand_start:cand_end+1], self.data[self.fp2_idx, cand_start:cand_end+1]
                    
                    corr = np.corrcoef(seg1_full, seg2_full)[0, 1] if len(seg1_full) > 1 else 0.0
                    corr_check = corr < h_corr_thresh
                    if self.verbose: print(f"      - Check [Correlation Fp1-Fp2]: {corr:.2f} (Threshold: <{h_corr_thresh:.2f}) -> {'✅' if corr_check else '❌'}")
                    if not corr_check: continue

                    fp1_peak, fp2_peak = (np.max(np.abs(seg1_full)) if len(seg1_full)>0 else 0), (np.max(np.abs(seg2_full)) if len(seg2_full)>0 else 0)
                    raw_amp_check = fp1_peak >= raw_ch_abs_peak_thresh_h and fp2_peak >= raw_ch_abs_peak_thresh_h
                    if self.verbose: print(f"      - Check [Raw channel peak]: Fp1={fp1_peak:.1f}uV, Fp2={fp2_peak:.1f}uV (Threshold: >={raw_ch_abs_peak_thresh_h:.1f}uV) -> {'✅' if raw_amp_check else '❌'}")
                    if not raw_amp_check: continue
                    
                    is_biphasic = False
                    if biphasic_check_enabled:
                        duration_ms = (cand_end - cand_start + 1) / self.freq * 1000.0
                        is_long_event = duration_ms > biphasic_skip_duration_ms
                        if self.verbose: print(f"      - Check [Biphasic - duration exemption]: Duration={duration_ms:.0f}ms (Threshold: >{biphasic_skip_duration_ms}ms) -> {'✅ Skip check' if is_long_event else '❌ Need to check'}")

                        if not is_long_event:
                            for ch_idx, ch_seg_full, ch_name in [(self.fp1_idx, seg1_full, 'Fp1'), (self.fp2_idx, seg2_full, 'Fp2')]:
                                if is_biphasic: break # If one channel is bad, no need to check the other

                                if self.verbose: print(f"      - Check [Biphasic @ {ch_name}]:")

                                if len(ch_seg_full) == 0: continue
                                main_peak_val = ch_seg_full[np.argmax(np.abs(ch_seg_full))]

                                # Check preceding wave
                                lookbehind_start = max(0, cand_start - int(lookbehind_ms * self.freq / 1000.0))
                                preceding_segment = self.data[ch_idx, lookbehind_start:cand_start]
                                if len(preceding_segment) > 1:
                                    opposite_peak_amp = -np.min(preceding_segment) if main_peak_val > 0 else np.max(preceding_segment)
                                    if opposite_peak_amp > preceding_neg_thresh_uv and opposite_peak_amp > abs(main_peak_val) * preceding_neg_ratio:
                                        if self._is_segment_sharp(preceding_segment, sharpness_thresh)[1]:
                                            is_biphasic = True

                                if is_biphasic: continue # Go to the next channel check (or exit if already found)

                                # Check succeeding wave
                                lookahead_start = cand_end + 1
                                lookahead_end = min(self.num_samples, lookahead_start + int(lookahead_ms * self.freq / 1000.0))
                                if lookahead_start < lookahead_end:
                                    succeeding_segment = self.data[ch_idx, lookahead_start:lookahead_end]
                                    if len(succeeding_segment) > 1:
                                        opposite_peak_amp = -np.min(succeeding_segment) if main_peak_val > 0 else np.max(succeeding_segment)
                                        if opposite_peak_amp > succeeding_neg_thresh_uv and opposite_peak_amp > abs(main_peak_val) * succeeding_neg_ratio:
                                            if self._is_segment_sharp(succeeding_segment, sharpness_thresh)[1]:
                                                is_biphasic = True
                    if self.verbose:
                        if is_biphasic: print(f"      - Check [Biphasic]: Biphasic waveform found in raw signal. -> ❌ Rejected")
                        else: print(f"      - Check [Biphasic]: No biphasic waveform found in raw signal. -> ✅ Passed")

                    if is_biphasic:
                        continue 

                    if self.verbose: print("      - Result: ⭐⭐⭐ HEOG_Fp event detected! ⭐⭐⭐")
                    event_segment_heog = heog_signal_fp_full[cand_start:cand_end+1]
                    direction_label = 'eog_left' if np.max(event_segment_heog) > -np.min(event_segment_heog) else 'eog_right'
                    overlapping_wins = [idx for idx, s_start in enumerate(self.start_indices) if max(s_start, cand_start) < min(s_start + self.window_size, cand_end)]
                    if overlapping_wins:
                        ch_indices_to_mark = [self.fp1_idx, self.fp2_idx]
                        for ch_idx in ch_indices_to_mark:
                            self.masks[direction_label][ch_idx, overlapping_wins] = True
                        self.precise_eog_events.append({'label': direction_label, 'ch_indices': ch_indices_to_mark, 
                                                        'start_sample': cand_start, 'end_sample': cand_end,
                                                        'window_index': i_win})
            # Validate HEOG_T events
            if self.f7_idx is not None and self.f8_idx is not None and 'heog_t' in self._global_eog_candidates:
                if self.verbose: print("\n" + "#"*30 + " HEOG (Horizontal Eye Movement @ F7/F8) Validation " + "#"*30)
                for i, candidate in enumerate(self._global_eog_candidates['heog_t']):
                    cand_start, cand_end = candidate['start'], candidate['end']
                    if max(start_sample, cand_start) >= min(end_sample, cand_end): continue
                    if self.verbose:
                        cand_start_s, cand_end_s = cand_start / self.freq, cand_end / self.freq
                        print(f"\n    --- HEOG_T Candidate #{i+1} (Absolute time: {cand_start_s:.3f}s-{cand_end_s:.3f}s) ---")

                    seg1_full, seg2_full = self.data[self.f7_idx, cand_start:cand_end+1], self.data[self.f8_idx, cand_start:cand_end+1]
                    corr = np.corrcoef(seg1_full, seg2_full)[0, 1] if len(seg1_full) > 1 else 0.0
                    corr_check = corr < h_corr_thresh
                    if self.verbose: print(f"      - Check [Correlation F7-F8]: {corr:.2f} (Threshold: <{h_corr_thresh:.2f}) -> {'✅' if corr_check else '❌'}")
                    if not corr_check: continue

                    f7_peak, f8_peak = (np.max(np.abs(seg1_full)) if len(seg1_full)>0 else 0), (np.max(np.abs(seg2_full)) if len(seg2_full)>0 else 0)
                    raw_amp_check = f7_peak >= raw_ch_abs_peak_thresh_h and f8_peak >= raw_ch_abs_peak_thresh_h
                    if self.verbose: print(f"      - Check [Raw channel peak]: F7={f7_peak:.1f}uV, F8={f8_peak:.1f}uV (Threshold: >={raw_ch_abs_peak_thresh_h:.1f}uV) -> {'✅' if raw_amp_check else '❌'}")

                    if not raw_amp_check: continue
                    
                    is_biphasic = False
                    if biphasic_check_enabled:
                        duration_ms = (cand_end - cand_start + 1) / self.freq * 1000.0
                        is_long_event = duration_ms > biphasic_skip_duration_ms
                        if self.verbose: print(f"      - Check [Biphasic Wave - Duration Exemption]: Duration = {duration_ms:.0f}ms (Threshold: >{biphasic_skip_duration_ms}ms) -> {'✅ Skipped check' if is_long_event else '❌ Needs to be checked'}")

                        if not is_long_event:
                            for ch_idx, ch_seg_full, ch_name in [(self.f7_idx, seg1_full, 'F7'), (self.f8_idx, seg2_full, 'F8')]:
                                if is_biphasic: break 

                                if self.verbose: print(f"      - Check [bipolar wave @ {ch_name}]:")

                                if len(ch_seg_full) == 0: continue
                                main_peak_val = ch_seg_full[np.argmax(np.abs(ch_seg_full))]
                                lookbehind_start = max(0, cand_start - int(lookbehind_ms * self.freq / 1000.0))
                                preceding_segment = self.data[ch_idx, lookbehind_start:cand_start]
                                if len(preceding_segment) > 1:
                                    opposite_peak_amp = -np.min(preceding_segment) if main_peak_val > 0 else np.max(preceding_segment)
                                    if opposite_peak_amp > preceding_neg_thresh_uv and opposite_peak_amp > abs(main_peak_val) * preceding_neg_ratio:
                                        if self._is_segment_sharp(preceding_segment, sharpness_thresh)[1]:
                                            is_biphasic = True

                                if is_biphasic: continue 
                                lookahead_start = cand_end + 1
                                lookahead_end = min(self.num_samples, lookahead_start + int(lookahead_ms * self.freq / 1000.0))
                                if lookahead_start < lookahead_end:
                                    succeeding_segment = self.data[ch_idx, lookahead_start:lookahead_end]
                                    if len(succeeding_segment) > 1:
                                        opposite_peak_amp = -np.min(succeeding_segment) if main_peak_val > 0 else np.max(succeeding_segment)
                                        if opposite_peak_amp > succeeding_neg_thresh_uv and opposite_peak_amp > abs(main_peak_val) * succeeding_neg_ratio:
                                            if self._is_segment_sharp(succeeding_segment, sharpness_thresh)[1]:
                                                is_biphasic = True
                    if self.verbose:
                        if is_biphasic: print(f"      - Check [Biphasic]: Biphasic waveform found in raw signal. -> ❌ Rejected")
                        else: print(f"      - Check [Biphasic]: No biphasic waveform found in raw signal. -> ✅ Passed")

                    if is_biphasic:
                        continue

                    if self.verbose: print("      - Result: ⭐⭐⭐ HEOG_F7 event detected! ⭐⭐⭐")
                    event_segment_heog = heog_signal_t_full[cand_start:cand_end+1]
                    direction_label = 'eog_left' if np.max(event_segment_heog) > -np.min(event_segment_heog) else 'eog_right'
                    overlapping_wins = [idx for idx, s_start in enumerate(self.start_indices) if max(s_start, cand_start) < min(s_start + self.window_size, cand_end)]
                    if overlapping_wins:
                        ch_indices_to_mark = [self.f7_idx, self.f8_idx]
                        for ch_idx in ch_indices_to_mark:
                            self.masks[direction_label][ch_idx, overlapping_wins] = True
                        self.precise_eog_events.append({'label': direction_label, 'ch_indices': ch_indices_to_mark, 
                                                        'start_sample': cand_start, 'end_sample': cand_end,
                                                        'window_index': i_win})

    def run_detection(self):

        self.masks = {name: np.zeros((self.num_channels, self.window_num), dtype=bool) 
                    for name in ["global_bad", "amp", "muscle", 
                                "eog_v", "eog_left", "eog_right", "drowsiness", 
                                "flat", "nan_inf", "respiration", "low_corr",
                                "severe_artifact", 
                                "eog_v_positive", "eog_v_veto"
                                ]}
        self.precise_eog_events = [] 
        if self.verbose: print("-> Step 1: Global Artifact & Prerequisite Calculation...")
        self.global_bad_channels_indices = self._detect_bad_channels_global(
            self.params['T_global_bad_ch_std_uV'], self.params['T_global_bad_ch_corr'], 
            self.params['max_global_bad_ch_fraction'], self._get_low_pass_filtered_data(cutoff_freq=self.params.get('bad_ch_corr_lpf_cutoff', 40.0)))
        for ch_idx in self.global_bad_channels_indices:
            self.masks["global_bad"][ch_idx, :] = True
        
        low_corr_mask_samples = self._detect_sliding_correlation_artifacts()
        if np.any(low_corr_mask_samples):
            for i_win, i_start in enumerate(self.start_indices):
                i_end = i_start + self.window_size
                window_low_corr_mask = np.mean(low_corr_mask_samples[:, i_start:i_end], axis=1) > 0.5
                self.masks['low_corr'][:, i_win] = window_low_corr_mask
            
        self.respiration_mask_timeseries = self._detect_respiration_morphological()
        if self.verbose: print("-> Step 2: Pre-calculating all window-based metrics...")
        self.rSDs = np.zeros((self.num_channels, self.window_num))
        self.roughnesses = np.zeros((self.num_channels, self.window_num))
        muscle_power_mask = np.zeros((self.num_channels, self.window_num), dtype=bool)
        volatility_mask = np.zeros((self.num_channels, self.window_num), dtype=bool)
        kurtosis_mask = np.zeros((self.num_channels, self.window_num), dtype=bool)
        sw_suppression_mask = np.ones((self.num_channels, self.window_num), dtype=bool)
        rhythmicity_mask = np.ones((self.num_channels, self.window_num), dtype=bool)

        debug_metrics = { 'power_ratios': np.zeros((self.num_channels, self.window_num)), 'muscle_rms': np.zeros((self.num_channels, self.window_num)), 'kurtosis': np.zeros((self.num_channels, self.window_num)), 'sw_ratio': np.zeros((self.num_channels, self.window_num)), 'concentration': np.zeros((self.num_channels, self.window_num)), }
        min_volatility_thresh = self.params.get("T_muscle_min_volatility_uV", 6.0)
        muscle_ratio_thresh = self.params['T_muscle_power_ratio']
        muscle_rms_thresh = self.params['T_muscle_min_uV_rms']
        max_kurtosis_thresh = self.params.get("T_muscle_max_kurtosis", 10.0)
        sw_suppression_enabled = self.params.get("T_muscle_slow_wave_suppression_enabled", True)
        rhythm_check_enabled = self.params.get("T_muscle_rhythmicity_check_enabled", True)
        muscle_band_data = self._get_band_pass_filtered_data(lowcut=30.0, highcut=99.0)
        if sw_suppression_enabled:
            sw_max_freq = self.params.get("T_muscle_slow_wave_max_freq_hz", 4.0); hf_to_sw_ratio = self.params.get("T_muscle_hf_to_slow_wave_amp_ratio", 0.20)
            nyquist = 0.5 * self.freq
            b, a = butter(4, sw_max_freq / nyquist, btype='low'); slow_wave_data = filtfilt(b, a, self.data, axis=1)
        if rhythm_check_enabled:
            rhythm_freq_band = self.params.get("T_muscle_rhythmicity_freq_band_hz", [4, 20]); spectral_conc_thresh = self.params.get("T_muscle_spectral_concentration_thresh", 8.0)

        for i_win, i_start in enumerate(self.start_indices):
            i_end = i_start + self.window_size
            window_data = self.data[:, i_start : i_end]
            
            self.rSDs[:, i_win] = _mat_iqr(window_data, axis=1) * IQR_TO_SD
            if window_data.shape[1] > 2: self.roughnesses[:, i_win] = np.sqrt(np.mean(np.diff(window_data, n=2, axis=1)**2, axis=1))
            
            flat_mask_window, nan_inf_mask_window = self._detect_flat_nan_inf(window_data)
            self.masks["flat"][:, i_win] = flat_mask_window
            self.masks["nan_inf"][:, i_win] = nan_inf_mask_window
            window_muscle_band_data = muscle_band_data[:, i_start : i_end]
            power_ratios = self._calculate_power_ratios(window_data)
            muscle_rms = np.sqrt(np.mean(window_muscle_band_data**2, axis=1))
            muscle_power_mask[:, i_win] = (power_ratios > muscle_ratio_thresh) & (muscle_rms > muscle_rms_thresh)
            debug_metrics['power_ratios'][:, i_win], debug_metrics['muscle_rms'][:, i_win] = power_ratios, muscle_rms
            
            volatility_mask[:, i_win] = np.std(window_data, axis=1) > min_volatility_thresh
            win_kurtosis = kurtosis(window_muscle_band_data, axis=1, fisher=True)
            kurtosis_mask[:, i_win] = win_kurtosis < max_kurtosis_thresh
            debug_metrics['kurtosis'][:, i_win] = win_kurtosis

            if sw_suppression_enabled:
                window_slow_wave_data = slow_wave_data[:, i_start : i_end]
                slow_wave_amp = np.std(window_slow_wave_data, axis=1)
                slow_wave_amp[slow_wave_amp < 1e-6] = 1e-6
                current_sw_ratio = muscle_rms / slow_wave_amp
                sw_suppression_mask[:, i_win] = current_sw_ratio >= hf_to_sw_ratio
                debug_metrics['sw_ratio'][:, i_win] = current_sw_ratio

            if rhythm_check_enabled:
                win_len = min(window_data.shape[1], self.freq)
                freqs, psd = welch(window_data, fs=self.freq, nperseg=win_len, axis=1)
                band_indices = np.where((freqs >= rhythm_freq_band[0]) & (freqs <= rhythm_freq_band[1]))[0]
                if len(band_indices) > 0:
                    concentration = np.divide(np.max(psd[:, band_indices], axis=1), np.mean(psd[:, band_indices], axis=1), out=np.zeros(self.num_channels), where=np.mean(psd[:, band_indices], axis=1)!=0)
                    rhythmicity_mask[:, i_win] = concentration < spectral_conc_thresh
                    debug_metrics['concentration'][:, i_win] = concentration
        self._detect_eog_and_drowsiness_v2()

        if self.verbose: print("\n-> Step 4: Z-score Analysis and Final Mask Combination...")
        winmask_bad_basic = self.masks["flat"] | self.masks["global_bad"] | self.masks["nan_inf"]
        
        mask_amp_spa = np.nan_to_num(self._calculate_spatial_zscores(self.rSDs, winmask_bad_basic)) > self.params['T_ampzsd']
        mask_amp_tem = np.nan_to_num(self._calculate_temporal_zscores(self.rSDs, winmask_bad_basic | mask_amp_spa)) > self.params['T_ampzsd']

        self._amp_mask_raw = mask_amp_spa | mask_amp_tem
        self._low_corr_mask_raw = self.masks.pop('low_corr', np.zeros((self.num_channels, self.window_num), dtype=bool))

        if 'amp' in self.masks:
            del self.masks['amp']


        zscores_muscle_rough = self._calculate_temporal_zscores(self.roughnesses, winmask_bad_basic)
        muscle_roughness_mask = np.nan_to_num(zscores_muscle_rough) > self.params['T_roughness_zsd']
        preliminary_muscle_mask = muscle_power_mask | muscle_roughness_mask
        final_muscle_mask = preliminary_muscle_mask & volatility_mask & kurtosis_mask & sw_suppression_mask & rhythmicity_mask
        self.masks["muscle"] = final_muscle_mask

        if self.verbose:
            print("\n--- [Debug Module] Detailed muscle detection decision process ---")
            debug_channel_name = self.params.get('debug_muscle_channel_of_interest')
            debug_channel_idx = self.ch_order.index(debug_channel_name) if debug_channel_name and debug_channel_name in self.ch_order else None
            
            if debug_channel_idx is not None:
                print(f"\n--- [Deep Debug Mode] Printing detailed report for channel {debug_channel_name} ---")
                for i_win in range(self.window_num):
                    power_check = muscle_power_mask[debug_channel_idx, i_win]
                    rough_check = muscle_roughness_mask[debug_channel_idx, i_win]
                    final_decision = final_muscle_mask[debug_channel_idx, i_win]
                    
                    pr_val = debug_metrics['power_ratios'][debug_channel_idx, i_win]
                    rms_val = debug_metrics['muscle_rms'][debug_channel_idx, i_win]
                    rough_z_val = zscores_muscle_rough[debug_channel_idx, i_win]
                    kurt_val = debug_metrics['kurtosis'][debug_channel_idx, i_win]
                    sw_ratio_val = debug_metrics['sw_ratio'][debug_channel_idx, i_win]
                    conc_val = debug_metrics['concentration'][debug_channel_idx, i_win]

                    print(f"  --- @ Window#{i_win} ---")
                    print(f"    - [Main] Power/RMS: Ratio={pr_val:.2f}>{muscle_ratio_thresh:.2f}({'✅' if pr_val>muscle_ratio_thresh else '❌'}) & RMS={rms_val:.1f}>{muscle_rms_thresh:.1f}uV({'✅' if rms_val>muscle_rms_thresh else '❌'}) -> {'✅' if power_check else '❌'}")
                    print(f"    - [Main] Roughness: Z-score={rough_z_val:.2f}>{self.params['T_roughness_zsd']:.1f} -> {'✅' if rough_check else '❌'}")
                    print(f"    - [Veto] Kurtosis: Kurt={kurt_val:.1f}<{max_kurtosis_thresh:.1f} -> {'✅' if kurtosis_mask[debug_channel_idx, i_win] else '❌'}")
                    print(f"    - [Veto] Slow wave suppression: Ratio={sw_ratio_val:.2f}>={hf_to_sw_ratio:.2f} -> {'✅' if sw_suppression_mask[debug_channel_idx, i_win] else '❌'}")
                    print(f"    - [Veto] Rhythmicity: Conc={conc_val:.1f}<{spectral_conc_thresh:.1f} -> {'✅' if rhythmicity_mask[debug_channel_idx, i_win] else '❌'}")
                    print(f"    - [Final Decision]: {'⭐⭐⭐ Muscle ⭐⭐⭐' if final_decision else '--- Not muscle ---'}")
            else:
                candidates_by_channel = {}
                for i_ch, i_win in np.argwhere(preliminary_muscle_mask):
                    if i_ch not in candidates_by_channel: candidates_by_channel[i_ch] = []
                    candidates_by_channel[i_ch].append(i_win)

                max_events_to_print_per_channel = 3
                for i_ch, win_indices in candidates_by_channel.items():
                    ch_name = self.ch_order[i_ch]
                    print(f"\n--- [Debug] {ch_name:<5} channel (found {len(win_indices)} suspected muscle windows) ---")
                    for i_win in win_indices[:max_events_to_print_per_channel]:
                        final_decision = final_muscle_mask[i_ch, i_win]
                        pr_val = debug_metrics['power_ratios'][i_ch, i_win]
                        rms_val = debug_metrics['muscle_rms'][i_ch, i_win]
                        rough_z_val = zscores_muscle_rough[i_ch, i_win]
                        power_check = muscle_power_mask[i_ch, i_win]
                        rough_check = muscle_roughness_mask[i_ch, i_win]
                        print(f"  --- [Candidate Event] @ Window#{i_win} ---")
                        print(f"    - [Main] Power/RMS: Ratio={pr_val:.2f}>{muscle_ratio_thresh:.2f}({'✅' if pr_val>muscle_ratio_thresh else '❌'}) & RMS={rms_val:.1f}>{muscle_rms_thresh:.1f}uV({'✅' if rms_val>muscle_rms_thresh else '❌'}) -> {'✅' if power_check else '❌'}")
                        print(f"    - [Main] Roughness: Z-score={rough_z_val:.2f}>{self.params['T_roughness_zsd']:.1f} -> {'✅' if rough_check else '❌'}")
                        print(f"    - [Final Decision]: {'⭐⭐⭐ Muscle ⭐⭐⭐' if final_decision else '--- Vetoed ---'}")

            print("\n  [Muscle detection summary]:")
            print(f"    - Total suspected muscle windows: {np.sum(preliminary_muscle_mask)}")
            print(f"    - After all veto conditions, final confirmed muscle windows: {np.sum(final_muscle_mask)}")
            print(f"    - Total vetoed windows: {np.sum(preliminary_muscle_mask) - np.sum(final_muscle_mask)}")
        if self.verbose:
            print("-> Step 5: Applying nan/inf boundary suppression to raw amplitude and correlation masks...")
        
        pre_mask_s = self.params.get("T_pre_nan_inf_eog_mask_s", 0.5)
        post_mask_s = self.params.get("T_post_nan_inf_eog_mask_s", 0.5)

        if pre_mask_s > 0 or post_mask_s > 0:
            pre_mask_windows = int(np.ceil(pre_mask_s / self.step_size_s))
            post_mask_windows = int(np.ceil(post_mask_s / self.step_size_s))
            
            for i_ch in range(self.num_channels):
                ch_nan_inf_mask = self.masks['nan_inf'][i_ch, :]
                if not np.any(ch_nan_inf_mask):
                    continue
                
                diff_mask = np.diff(np.concatenate(([0], ch_nan_inf_mask.astype(int), [0])))
                onset_windows = np.where(diff_mask == 1)[0]
                offset_windows = np.where(diff_mask == -1)[0]

                suppression_mask_ch = np.zeros_like(ch_nan_inf_mask, dtype=bool)
                if pre_mask_windows > 0:
                    for start_artifact in onset_windows:
                        start_suppress = max(0, start_artifact - pre_mask_windows)
                        suppression_mask_ch[start_suppress:start_artifact] = True
                
                if post_mask_windows > 0:
                    for end_artifact in offset_windows:
                        start_suppress = end_artifact
                        end_suppress = start_suppress + post_mask_windows
                        suppression_mask_ch[start_suppress:end_suppress] = True
                
                self._amp_mask_raw[i_ch, suppression_mask_ch] = False
                self._low_corr_mask_raw[i_ch, suppression_mask_ch] = False
                self.masks['muscle'][i_ch, suppression_mask_ch] = False
        self._apply_nan_inf_boundary_suppression()
        self.masks['eog_v'] = self.masks['eog_v_positive'] & ~self.masks['eog_v_veto']
        
        
        if self.verbose: print("\nArtifact detection finished.")

    def _postprocess_mask(self, mask: np.ndarray, iterations: int) -> np.ndarray:
        if iterations == 0 or not np.any(mask): return mask
        structure = np.ones((1, 3)); return binary_closing(mask, structure=structure, iterations=iterations)


    def _get_low_pass_filtered_data(self, cutoff_freq: float, order: int = 4) -> np.ndarray:
        nyquist = 0.5 * self.freq; normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False); return filtfilt(b, a, self.data, axis=1)

    def _detect_bad_channels_global(self, T_std_uV, T_corr, max_fraction, filtered_data: np.ndarray) -> List[int]:

        bad_by_std = {i for i, std in enumerate(np.std(self.data, axis=1)) if not (T_std_uV[0] < std < T_std_uV[1])}
        
        if self.verbose:
            print("\n--- [Debug] Starting global bad channel detection ---")
            print(f"  - Std threshold range: {T_std_uV[0]} uV < std < {T_std_uV[1]} uV")
            print(f"  - Neighbor correlation threshold: mean(abs(corr)) < {T_corr}")
            print("-" * 35)
            print("  [Stage 1: Std-based detection]")
            for i, std in enumerate(np.std(self.data, axis=1)):
                ch_name = self.ch_order[i]
                if i in bad_by_std:
                    decision = "❌ Bad channel"
                    reason = "Flat line" if std < T_std_uV[0] else "Noise"
                    print(f"    - Channel '{ch_name}': std = {std:.2f} uV -> {decision} (Reason: {reason})")
                else:
                    decision = "✅ Passed"
                    print(f"    - Channel '{ch_name}': std = {std:.2f} uV -> {decision}")

        bad_indices = bad_by_std

        NEIGHBORS = {'Fp1':['Fp2','F7','F3'],'Fp2':['Fp1','F8','F4'],'F7':['Fp1','T3','F3'],'F3':['Fp1','F7','C3','Fz'],'Fz':['F3','F4'],'C3':['F3','T3','P3','Cz'],'P3':['C3','T5','O1','Pz'],'O1':['P3','O2','T5'],'O2':['P4','O1','T6'],'F4':['Fp2','Fz','C4','F8'],'C4':['F4','T4','P4','Cz'],'P4':['C4','T6','O2','Pz'],'F8':['Fp2','F4','T4'],'T3':['F7','C3','T5'],'T4':['F8','C4','T6'],'T5':['T3','P3','O1'],'T6':['T4','P4','O2'],'Cz':['C3','C4'],'Pz':['P3','P4']}
        good_ch_indices = [i for i in range(self.num_channels) if i not in bad_indices]

        if self.verbose:
            print("\n  [Stage 2: Neighbor correlation-based detection]")

        bad_by_corr = set()
        for i in range(self.num_channels):
            if i in bad_indices: 
                if self.verbose: print(f"    - Channel '{self.ch_order[i]}': Skipped (already bad channel)")
                continue
            ch_name = self.ch_order[i]
            if ch_name not in NEIGHBORS: 
                if self.verbose: print(f"    - Channel '{self.ch_order[i]}': Skipped (no neighbor defined)")
                continue
            
            neighbor_indices = [self.ch_order.index(n) for n in NEIGHBORS[ch_name] if n in self.ch_order and self.ch_order.index(n) in good_ch_indices]
            if not neighbor_indices: 
                if self.verbose: print(f"    - Channel '{self.ch_order[i]}': Skipped (no available neighbor)")
                continue
            
            neighbor_names = [self.ch_order[n_idx] for n_idx in neighbor_indices]
            corrs = [np.corrcoef(filtered_data[i], filtered_data[n_idx])[0, 1] for n_idx in neighbor_indices]
            mean_abs_corr = np.mean(np.abs(corrs))
            
            if mean_abs_corr < T_corr: 
                bad_by_corr.add(i)
                decision = "❌ Bad channel"
            else:
                decision = "✅ Passed"

            if self.verbose:
                print(f"    - Channel '{ch_name}': mean correlation with neighbors {neighbor_names} = {mean_abs_corr:.3f} -> {decision}")

        bad_indices.update(bad_by_corr)

        if len(bad_indices) / self.num_channels > max_fraction: 
            if self.verbose: 
                print("\n  [Stage 3: Final decision]")
                print(f"    - Warning: Found {len(bad_indices)} bad channels ({len(bad_indices)/self.num_channels:.0%}), exceeds limit of {max_fraction:.0%}.")
                print("    - Decision: Disable global bad channel detection, return empty list.")
            return []
        
        if self.verbose:
            print("\n--- [Debug] Global bad channel detection finished ---")
            final_bad_names = [self.ch_order[i] for i in sorted(list(bad_indices))]
            print(f"  -> Final bad channel list: {final_bad_names}")
                
        return list(bad_indices)



    def _detect_flat_nan_inf(self, window_data:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T_flat, T_range = self.params['T_flat'], self.params['T_range']

        flat_by_mad = _mad(window_data, axis=1) < T_flat
        flat_by_sd = np.std(window_data, axis=1) < T_flat
        flat_mask = flat_by_mad & flat_by_sd

        bad_by_nan = np.isnan(window_data).any(axis=1)
        bad_by_inf = np.isinf(window_data).any(axis=1)
        out_of_range = (window_data > T_range[1]).any(axis=1) | (window_data < T_range[0]).any(axis=1)
        nan_inf_range_mask = bad_by_nan | bad_by_inf | out_of_range

        return flat_mask, nan_inf_range_mask

    def _calculate_spatial_zscores(self, metrics:np.ndarray, bad_mask:np.ndarray, debug_label:str="") -> np.ndarray:
        zscores = np.full_like(metrics, np.nan, dtype=float)
        for i_win in range(self.window_num):
            win_mask_good = ~bad_mask[:, i_win]
            if np.sum(win_mask_good) < 3: continue
            
            win_metrics = metrics[win_mask_good, i_win]
            median = np.median(win_metrics)
            rsd = _mat_iqr(win_metrics) * IQR_TO_SD

            if self.verbose and debug_label == "amp_spa":
                print(f"  [Debug-amp_spa] Window #{i_win}: Spatial statistics -> Median={median:.2f}, RSD={rsd:.2f}")

            if rsd > 1e-9: 
                zscores[win_mask_good, i_win] = (win_metrics - median) / rsd
                
        return zscores

    def _calculate_temporal_zscores(self, metrics:np.ndarray, bad_mask:np.ndarray, debug_label:str="") -> np.ndarray:
        zscores = np.full_like(metrics, np.nan, dtype=float)
        for i_ch in range(self.num_channels):
            ch_mask_good = ~bad_mask[i_ch, :]
            if np.sum(ch_mask_good) < 5: continue
            
            ch_metrics = metrics[i_ch, ch_mask_good]
            median = np.median(ch_metrics)
            rsd = _mat_iqr(ch_metrics) * IQR_TO_SD

            if self.verbose and debug_label == "amp_tem":
                print(f"  [Debug-amp_tem] Channel '{self.ch_order[i_ch]}': Temporal statistics -> Median={median:.2f}, RSD={rsd:.2f}")
                
            if rsd > 1e-9: 
                zscores[i_ch, ch_mask_good] = (ch_metrics - median) / rsd
                
        return zscores


    def _calculate_power_ratios(self, window_data: np.ndarray) -> np.ndarray:
        return np.array([self._calculate_band_power_ratio(ch_data) for ch_data in window_data])

    def _calculate_band_power_ratio(self, data, muscle_band=[30, 99], total_band=[1, 40]):
        win_len = min(len(data), self.freq); freqs, psd = welch(data, fs=self.freq, nperseg=win_len, scaling='density')
        muscle_power = np.sum(psd[(freqs >= muscle_band[0]) & (freqs <= muscle_band[1])])
        total_power = np.sum(psd[(freqs >= total_band[0]) & (freqs <= total_band[1])])
        return muscle_power / total_power if total_power > 1e-10 else 0.0

    def get_annotations_dict(self, source_file_path: str = "N/A", global_start_time: float = 0.0) -> dict:
        if self.verbose:
            print("\n--- [get_annotations_dict] Stage 1: Start composing 'severe_artifact'...")

        def _convert_mask_to_bboxes(mask, wave_type, ch_indices):
            bboxes = []
            for i_ch in ch_indices:
                if not np.any(mask[i_ch, :]): continue
                diff_mask = np.diff(np.concatenate(([0], mask[i_ch, :].astype(int), [0])))
                onset_wins = np.where(diff_mask == 1)[0]
                offset_wins = np.where(diff_mask == -1)[0]
                for on, off in zip(onset_wins, offset_wins):
                    start_s = self.start_indices[on] / self.freq
                    end_s = (self.start_indices[off - 1] + self.window_size) / self.freq
                    bboxes.append({
                        "channel_idx": i_ch,
                        "time_range": [round(start_s, 3), round(end_s, 3)],
                        "wave_type": wave_type
                    })
            return bboxes
        all_ch_indices = list(range(self.num_channels))
        amp_bboxes_raw = _convert_mask_to_bboxes(getattr(self, '_amp_mask_raw', np.zeros((self.num_channels, self.window_num))), 'amp', all_ch_indices)
        low_corr_bboxes_raw = _convert_mask_to_bboxes(getattr(self, '_low_corr_mask_raw', np.zeros((self.num_channels, self.window_num))), 'low_corr', all_ch_indices)
        final_severe_artifact_bboxes = []
        abs_amp_thresh = self.params.get("T_severe_artifact_abs_amp_thresh_uv", 200.0)
        remaining_amp_bboxes = []
        for bbox in amp_bboxes_raw:
            ch_idx, start_time, end_time = bbox['channel_idx'], bbox['time_range'][0], bbox['time_range'][1]
            start_sample, end_sample = int(start_time * self.freq), int(end_time * self.freq)
            segment_data = self.data[ch_idx, start_sample:end_sample]
            if segment_data.size > 0 and np.max(np.abs(segment_data)) > abs_amp_thresh:
                bbox['wave_type'] = 'severe_artifact'
                final_severe_artifact_bboxes.append(bbox)
            else:
                remaining_amp_bboxes.append(bbox)
        overlap_threshold_s = self.params.get("T_severe_artifact_min_duration_s", 0.5)
        consumed_ids_for_overlap = set()
        for box1 in remaining_amp_bboxes:
            for box2 in low_corr_bboxes_raw:
                if box1['channel_idx'] != box2['channel_idx']: continue
                overlap_start = max(box1['time_range'][0], box2['time_range'][0])
                overlap_end = min(box1['time_range'][1], box2['time_range'][1])
                if (overlap_end - overlap_start) > overlap_threshold_s:
                    ptp_thresh = self.params.get("T_severe_artifact_min_ptp_uv", 50.0)

                    ch_idx = box1['channel_idx']
                    start_sample = int(overlap_start * self.freq)
                    end_sample = int(overlap_end * self.freq)
                    segment_data = self.data[ch_idx, start_sample:end_sample]

                    if segment_data.size > 0:
                        ptp_amplitude = np.ptp(segment_data)
                        if ptp_amplitude > ptp_thresh:
                            new_severe_box = {"channel_idx": ch_idx, "time_range": [round(overlap_start, 3), round(overlap_end, 3)], "wave_type": "severe_artifact", "confidence": 1.0}
                            final_severe_artifact_bboxes.append(new_severe_box)
                            consumed_ids_for_overlap.add((box1['channel_idx'], box1['time_range'][0]))
                            consumed_ids_for_overlap.add((box2['channel_idx'], box2['time_range'][0]))

        if self.verbose:
            print(f"--- [get_annotations_dict] Stage 1 completed: Synthesize {len(final_severe_artifact_bboxes)} 'severe_artifact' annotations.")
            print(f"--- [get_annotations_dict] Stage 2: Start processing all other regular artifacts...")

        final_window_masks = self.get_all_masks_v2()
        other_bboxes = []
        eog_labels_to_skip = {'eog_v', 'eog_left', 'eog_right'}
        for label in final_window_masks.keys():
            if label in eog_labels_to_skip:
                continue
            if label != 'severe_artifact': 
                other_bboxes.extend(_convert_mask_to_bboxes(final_window_masks[label], label, all_ch_indices))
        
    # --- [Core modification]: Create nan_inf suppression region before processing precise_eog_events ---
        if hasattr(self, 'precise_eog_events') and self.precise_eog_events:
            if self.verbose:
                print("--- [get_annotations_dict] Stage 3: Apply final nan/inf suppression filtering to precise EOG events ---")

            nan_inf_suppression_ranges = defaultdict(list)
            pre_mask_s = self.params.get("T_pre_nan_inf_eog_mask_s", 0.5)
            post_mask_s = self.params.get("T_post_nan_inf_eog_mask_s", 0.5)
            
            nan_inf_bboxes = _convert_mask_to_bboxes(self.masks['nan_inf'], 'nan_inf', all_ch_indices)
            for bbox in nan_inf_bboxes:
                ch_idx = bbox['channel_idx']
                start, end = bbox['time_range']
                suppress_start = start - pre_mask_s
                suppress_end = end + post_mask_s
                nan_inf_suppression_ranges[ch_idx].append([suppress_start, suppress_end])
            
            # Merge suppression intervals for each channel
            for ch_idx in nan_inf_suppression_ranges:
                nan_inf_suppression_ranges[ch_idx] = self._merge_bboxes(
                    [{'channel_idx': ch_idx, 'wave_type': 'temp', 'time_range': r} for r in nan_inf_suppression_ranges[ch_idx]], 
                    max_gap_s=0.01
                )

            # 2. Iterate and filter precise_eog_events
            for event in self.precise_eog_events:
                start_sample, end_sample = event['start_sample'], event['end_sample']
                event_start_s, event_end_s = start_sample / self.freq, end_sample / self.freq
                label, ch_indices = event['label'], event['ch_indices']
                
                is_suppressed = False
                for ch_idx in ch_indices:
                    suppress_bboxes = nan_inf_suppression_ranges.get(ch_idx, [])
                    for s_bbox in suppress_bboxes:
                        if calculate_iou([event_start_s, event_end_s], s_bbox['time_range']) > 0:
                            is_suppressed = True
                            break
                    if is_suppressed:
                        break
                
                if is_suppressed:
                    if self.verbose:
                        print(f"  - [Suppression] Skipping '{label}' event at {event_start_s:.2f}s-{event_end_s:.2f}s because it is close to nan_inf.")
                    continue 
                if 'window_index' not in event: continue 
                win_idx = event['window_index']
                if win_idx >= self.window_num: continue
                main_ch_idx = ch_indices[0]
                if final_window_masks.get(label, np.zeros((self.num_channels, self.window_num), dtype=bool))[main_ch_idx, win_idx]:
                    for ch_idx in ch_indices:
                        other_bboxes.append({"channel_idx": ch_idx, "time_range": [float(round(event_start_s, 3)), float(round(event_end_s, 3))], "wave_type": label, "confidence": 1.0})

        if hasattr(self, 'respiration_mask_timeseries') and np.any(self.respiration_mask_timeseries):
            diff_mask = np.diff(np.concatenate(([0], self.respiration_mask_timeseries.astype(int), [0])))
            onset_samples= np.where(diff_mask == 1)[0]
            offset_samples = np.where(diff_mask == -1)[0]
            good_ch_indices = [i for i in range(self.num_channels) if i not in self.global_bad_channels_indices]
            for start_sample, end_sample in zip(onset_samples, offset_samples):
                for ch_idx in good_ch_indices:
                    start_time = round(global_start_time + start_sample / self.freq, 3)
                    end_time = round(global_start_time + end_sample / self.freq, 3)
                    other_bboxes.append({"channel_idx": ch_idx, "time_range": [float(start_time), float(end_time)], "wave_type": 'respiration', "confidence": 1.0})
        min_muscle_dur_final = self.params.get("T_muscle_final_min_duration_s", 0.5)
        
        final_bboxes = []
        for bbox in other_bboxes:
            if bbox['wave_type'] == 'muscle':
                duration = bbox['time_range'][1] - bbox['time_range'][0]
                if duration >= min_muscle_dur_final:
                    final_bboxes.append(bbox)
            else:
                final_bboxes.append(bbox)

        max_gap_s = self.params.get("T_eog_merge_max_gap_s", 0.01)
        merged_bboxes = self._merge_bboxes(final_bboxes, max_gap_s) 
        merged_bboxes.sort(key=lambda x: (x['time_range'][0], x['channel_idx']))
        final_all_bboxes = final_severe_artifact_bboxes + merged_bboxes
        final_all_bboxes.sort(key=lambda x: (x['time_range'][0], x['channel_idx']))

        total_duration = self.num_samples / self.freq
        result_dict = {
            "source_file": source_file_path, "global_start": global_start_time,
            "global_end": global_start_time + total_duration,
            "channels": {"ear_ref": None, "ave_ref": self.ch_order},
            "bboxes": final_all_bboxes,
            "state": None,
            "description": "Artifacts detected by EEGArtifact (Separated Logic)",
            "is_expert_annotated": False
        }
        return result_dict


def load_eeg_data(file_path: str, file_type: str, sfreq: Optional[int], ch_names_str: Optional[str]) -> Tuple[np.ndarray, int, list, str]:
    """Loads EEG data from H5 or NPY files."""
    print(f"-> Loading data from {file_path}...")
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    if file_type == 'h5':
        with h5py.File(file_path, 'r') as f:
            group_key = list(f.keys())[0]
            subject_group = f[group_key]
            eeg_dataset = subject_group['eeg']
            eeg_data_volts = eeg_dataset[:]
            loaded_sfreq = subject_group.attrs['sFreq']
            loaded_ch_names = [name.decode('utf-8') for name in eeg_dataset.attrs['chOrder']]
            eeg_data_microvolts = eeg_data_volts * 1e6
    elif file_type == 'npy':
        if sfreq is None or ch_names_str is None:
            raise ValueError("--sfreq and --ch-names are required for 'npy' file type.")
        eeg_data_microvolts = np.load(file_path)
        loaded_sfreq = sfreq
        loaded_ch_names = [ch.strip() for ch in ch_names_str.split(',')]
        if eeg_data_microvolts.shape[0] != len(loaded_ch_names):
            raise ValueError(f"Number of channels in data ({eeg_data_microvolts.shape[0]}) does not match number of channel names provided ({len(loaded_ch_names)}).")
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Choose 'h5' or 'npy'.")
    
    print(f"  - Loaded {eeg_data_microvolts.shape[1] / loaded_sfreq:.2f} seconds of data with {eeg_data_microvolts.shape[0]} channels.")
    return eeg_data_microvolts, loaded_sfreq, loaded_ch_names, str(path_obj.resolve())




import matplotlib.patches as mpatches


def visualize_artifacts_interactive(eeg_data: np.ndarray, sfreq: int, ch_names: list,
                                      model: EEGArtifact,
                                      output_folder: Path, segment_id: str,
                                      show_interactive: bool = False):

    print(f"-> Generating final visualization for {segment_id} ...")

    result_dict = model.get_annotations_dict(source_file_path=f"{segment_id}.h5", global_start_time=0.0)
    final_bboxes = result_dict.get('bboxes', [])

    display_channel_width = 150
    num_channels, num_samples = eeg_data.shape
    time_axis = np.arange(num_samples) / sfreq
    fig, ax = plt.subplots(figsize=(12, 16))
    fig.suptitle(f"Artifact Visualization: {segment_id} (Final Version)", fontsize=16)
    offset = np.arange(num_channels, 0, -1) * display_channel_width
    for i in range(num_channels):
        ax.plot(time_axis, -eeg_data[i] + offset[i], label=ch_names[i], color='black', linewidth=0.8)
    
    ax.set_xticks(np.arange(0, time_axis[-1] + 1, 1))
    ax.set_yticks(offset)
    ax.set_yticklabels(ch_names)
    ax.set_ylim(display_channel_width//2, num_channels * display_channel_width + display_channel_width//2)
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle=':', linewidth=0.5, axis='x')

    ANNOTATION_COLOR_MAP = {
        "severe_artifact": 'magenta',"nan_inf": 'black', "global_bad": 'gray',
        "flat": 'deepskyblue', "muscle": 'purple', 
        "eog_v": 'red', "eog_left": 'tomato', "eog_right": 'gold', "respiration": 'brown',
        "drowsiness": 'saddlebrown',
    }

    present_wave_types = sorted(list(set(b['wave_type'] for b in final_bboxes)))

    for bbox in final_bboxes:
        ch_idx, wave_type = bbox['channel_idx'], bbox['wave_type']
        start_sec, end_sec = bbox['time_range']
        
        color = ANNOTATION_COLOR_MAP.get(wave_type, 'cyan')
        
        if start_sec == end_sec and wave_type == 'ecg':
            line_y_start = offset[ch_idx] - display_channel_width / 2.5
            line_y_end = offset[ch_idx] + display_channel_width / 2.5
            ax.plot([start_sec, start_sec], [line_y_start, line_y_end], color=color, linestyle='--', linewidth=1.5, zorder=10)
        else:
            width_sec = end_sec - start_sec
            rect_y = offset[ch_idx] - display_channel_width / 2.5
            rect_height = display_channel_width / 2.5 * 2
            rect = mpatches.Rectangle((start_sec, rect_y), width_sec, rect_height,
                                      linewidth=1.5, linestyle='-',
                                      edgecolor=color, fill=False, zorder=10)
            ax.add_patch(rect)

    patches = [mpatches.Patch(color=ANNOTATION_COLOR_MAP.get(wt, 'cyan'), label=wt) for wt in present_wave_types]
    ax.legend(handles=patches, loc='upper right')

    save_path = output_folder / f"{segment_id}_visualization.png"
    plt.savefig(save_path, dpi=150)
    
    if show_interactive:
        plt.show(block=True)
    
    plt.close(fig)
    print(f"     - New style precise visualization image saved to: {save_path}")
import warnings 
if __name__ == '__main__':
    warnings.filterwarnings('error', category=RuntimeWarning)
    parser = argparse.ArgumentParser(description="EEG Artifact Detection (V11) - Batch Runner")
    parser.add_argument('--input-folder', required=True, type=str, help='Path to the input folder containing EEG data files.')
    parser.add_argument('--output-file', default='all_artifacts_results.json', type=str, help='Path to the single output JSON file.')
    parser.add_argument('--file-type', required=True, type=str, choices=['h5', 'npy'], help="Type of the input files to process.")
    
    # Optional arguments
    parser.add_argument('--sfreq', type=int,default=200, help='Sampling frequency. Required for .npy files.')
    parser.add_argument('--ch-names', type=str, default=",".join(CH_NAMES_DEFAULT), help='Comma-separated channel names.')
    
    # --- START: Visualization related parameter modification ---
    parser.add_argument('--no-visualize', action='store_true', help='Disable all visualization (no plot files saved).')
    parser.add_argument('--show-plot', action='store_true', help='Attempt to show an interactive plot window (will still save the file).')
    # --- END: Visualization related parameter modification ---
    
    parser.add_argument('--split-threshold', default=20.0, type=float, help='Duration in seconds to start chunking.')
    parser.add_argument('--chunk-duration', default=10.0, type=float, help='Duration of each chunk in seconds.')
    
    args = parser.parse_args()

    try:
        input_folder = Path(args.input_folder)
        files_to_process = list(input_folder.glob(f'*.{args.file_type}'))
        
        if not files_to_process:
            print(f"Error: No '.{args.file_type}' files found in: {input_folder}")
            sys.exit(1)
        
        print(f"Found {len(files_to_process)} files to process.")
        master_results = {}

        for i, file_path in enumerate(files_to_process):
            print(f"\n{'='*20} Processing file {i+1}/{len(files_to_process)}: {file_path.name} {'='*20}")
            eeg_data_microvolts, sfreq, ch_names, source_file_path = load_eeg_data(str(file_path), args.file_type, args.sfreq, args.ch_names)
            total_duration_s = eeg_data_microvolts.shape[1] / sfreq
            chunks_to_process = []
            file_basename = file_path.stem
            
            if total_duration_s > args.split_threshold:
                num_chunks = int(np.ceil(total_duration_s / args.chunk_duration))
                print(f"-> Data duration ({total_duration_s:.2f}s) > threshold. Splitting into {num_chunks} chunks.")
                chunk_len_samples = int(args.chunk_duration * sfreq)
                for j in range(num_chunks):
                    start_s, end_s = j * args.chunk_duration, (j + 1) * args.chunk_duration
                    start_sample, end_sample = j * chunk_len_samples, (j + 1) * chunk_len_samples
                    chunk_data = eeg_data_microvolts[:, start_sample:end_sample]
                    if chunk_data.shape[1] < sfreq * 1.0: continue
                    chunks_to_process.append({
                        "data": chunk_data, "segment_id": f"{file_basename}_seg{j+1}",
                        "global_start": start_s, "global_end": min(end_s, total_duration_s)
                    })
            else:
                chunks_to_process.append({
                    "data": eeg_data_microvolts, "segment_id": f"{file_basename}_seg1",
                    "global_start": 0.0, "global_end": total_duration_s
                })

            current_file_results = {}
            for chunk_info in chunks_to_process:
                print(f"---- Processing Segment: {chunk_info['segment_id']} ----")
                model = EEGArtifact(data=chunk_info["data"], freq=sfreq, ch_order=ch_names, verbose=False,debug_muscle_channel_of_interest='T6')
                final_masks = model.get_all_masks_v2()
                if not final_masks:
                    print("     - No artifacts detected or data was too short.")
                    continue
                json_output_dict =model.get_annotations_dict(
                    source_file_path=source_file_path,
                    global_start_time=chunk_info['global_start']
                )
                if json_output_dict.get('bboxes'):
                    json_output = {chunk_info['segment_id']: json_output_dict}
                    current_file_results.update(json_output)

                if not args.no_visualize:
                    output_path = Path(args.output_file)
                    vis_output_folder = output_path.parent / "visualizations"
                    vis_output_folder.mkdir(exist_ok=True)
                    visualize_artifacts_interactive(
                    eeg_data=chunk_info["data"], 
                    sfreq=sfreq, 
                    ch_names=ch_names, 
                    model=model,  
                    output_folder=vis_output_folder,
                    segment_id=chunk_info['segment_id'],
                    show_interactive=args.show_plot
                )


            if current_file_results:
                master_results[file_path.name] = current_file_results
                print(f"-> Finished processing '{file_path.name}', results aggregated.")
        
        if master_results:
            print(f"\n-> All files processed. Saving aggregated results to '{args.output_file}'...")
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(master_results, f, indent=4)
            print("   - Save complete.")
        else:
            print("\n-> No artifacts were detected across all files. No output file was created.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Batch artifact detection process finished. ---")