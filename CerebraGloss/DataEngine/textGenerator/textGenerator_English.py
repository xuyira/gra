'''
Rule-based text description generation.
API:
text = text_generator(YOLO_bboxes, artifact_bboxes, dominant_freq_matrix, max_amp_matrix, channel_list)
'''
import numpy as np
from collections import defaultdict
import math
import warnings
import re

#######################################
# Helper Function to Realize Core API #
#######################################
def calculate_iou(range1: list[float], range2: list[float]) -> float:
    """
    Calculate the intersection over union (IoU) of two time intervals.

    Args:
        range1/2 (list[float]): Time interval like [0.2, 1.5]
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

def merge_time_ranges(ranges: list[list[float]]) -> list[list[float]]:
    """
    Merge overlapping or adjacent time intervals.
    For example, [[2.0, 4.0], [3.0, 5.0]] merges into [[2.0, 5.0]].
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0]) 
    merged = [sorted_ranges[0]]
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:  
            merged[-1] = [last_start, max(last_end, current_end)]
        else:
            merged.append([current_start, current_end])
    return merged

def format_time_ranges_string(ranges: list[list[float]]) -> str:
    """
    Format a list of time intervals into a readable string, e.g. "during 2.0-5.0s, 6.4-6.6s".
    """
    parts = [f"{start:.1f}-{end:.1f}s" for start, end in ranges]
    return "during " + ", ".join(parts)

def format_duration_description(
    ranges: list[list[float]], 
    total_duration: float, 
    threshold: float = 0.8
) -> str:
    """
    Format time description based on the ratio of event duration to total recording duration.
    If the event occupies most of the time, return "throughout the recording period"; otherwise, return the specific time range.
    Args:
        ranges (list[list[float]]): List of event time intervals.
        total_duration (float): Total recording duration (seconds).
        threshold (float): Threshold to determine "throughout the period".
    Returns:
        str: Formatted time description string.
    """
    merged_ranges = merge_time_ranges(ranges)
    event_duration = sum(end - start for start, end in merged_ranges)
    
    if event_duration / total_duration >= threshold:
        return "throughout the recording period"
    else:
        return format_time_ranges_string(merged_ranges)

# Mapping from channel to brain region
CHANNEL_TO_LOBE = {
    'FP1': 'frontal', 'FP2': 'frontal', 'F3': 'frontal', 'F4': 'frontal', 'FZ': 'frontal',
    'C3': 'central', 'C4': 'central', 'CZ': 'central',
    'P3': 'parietal', 'P4': 'parietal', 'PZ': 'parietal',
    'T3': 'temporal', 'T4': 'temporal', 'T5': 'temporal', 'T6': 'temporal', 'F7': 'temporal', 'F8': 'temporal',
    'O1': 'occipital', 'O2': 'occipital',
}

# Mapping from channel to hemisphere
CHANNEL_TO_HEMISPHERE = {
    # left hemisphere (odd numbers)
    'FP1': 'left', 'F3': 'left', 'C3': 'left', 'P3': 'left', 'O1': 'left',
    'F7': 'left', 'T3': 'left', 'T5': 'left',
    # right hemisphere (even numbers)
    'FP2': 'right', 'F4': 'right', 'C4': 'right', 'P4': 'right', 'O2': 'right',
    'F8': 'right', 'T4': 'right', 'T6': 'right',
    # midline
    'FZ': 'midline', 'CZ': 'midline', 'PZ': 'midline',
}

# Wave type English name mapping
WAVE_TYPE_MAP = {
    'spike': 'spikes',
    'spsw': 'spike-and-slow-wave complexes',
    'sharp': 'sharps',
    'spindle': 'sleep spindles',
    'Kcomplex': 'K-complexes',
    'eyem': 'positive sharps/spikes'
}

def get_channel_lobe(channel_name: str) -> str:
    """Get the main brain region of the channel"""
    base_channel = channel_name.split('-')[0] # drop reference part
    return CHANNEL_TO_LOBE.get(base_channel, 'Unknown region')

def get_channel_hemisphere(channel_name: str) -> str:
    """
    Get the hemisphere of the channel.
    Return "left", "right" or "midline".
    """
    base_channel = channel_name.split('-')[0]
    return CHANNEL_TO_HEMISPHERE.get(base_channel, 'unknown')

def get_freq_band_name(freq: float) -> str:
    """Return the frequency band name according to the frequency value"""
    if freq < 4: return "Delta"
    elif freq < 8: return "Theta"
    elif freq < 14: return "Alpha"
    elif freq < 30: return "Beta"
    else: return "Gamma"

def get_amp_category(amp: float) -> str:
    """
    Return the amplitude category, i.e. low/medium/high amplitude
    """
    if amp < 25: return "low"
    elif amp <= 70: return "medium"
    else: return "high"

def summarize_channel_locations(channels: list[str], channel_list: list[str]) -> str:
    """
    Intelligently summarize the spatial distribution based on a set of channel names.
    1. Focal event (<=3 channels): Directly return channel names, e.g. "F3, F4".
    2. Widespread event (>=80% channels): Return "in the whole brain".
    3. Unilateral event: Determine if all brain regions in the hemisphere are covered to simplify description.
        - Full coverage -> "in the right hemisphere"
        - Partial coverage -> "in the right frontal and temporal lobes"
    4. Bilateral event: Use "bilateral" only when both sides are evenly distributed.
    5. Asymmetric event: Describe both sides separately, e.g. "in the left hemisphere and right frontal lobe".
    Return example: 'in the ...'
    """
    if not channels:
        return ""
    
    order_dict = {channel: idx for idx, channel in enumerate(channel_list)}
    unique_channels = sorted(list(set(channels)), key=lambda x: order_dict[x])
    num_event_channels = len(unique_channels)

    # 1. Prefer to handle very focal events (<=3 channels), directly list channel names for more precision
    if num_event_channels <= 3:
        return "in " + ", ".join(unique_channels)

    total_channels = len(channel_list)
    
    # 2. Whole brain judgment
    if num_event_channels >= total_channels * 0.8:
        return "in the whole brain"

    # 3. Detailed statistics by hemisphere and brain region
    lobes_by_hemisphere = defaultdict(set)
    # Pre-calculate all possible lobes for each hemisphere
    all_possible_lobes = defaultdict(set)
    for ch in channel_list:
        hemi = get_channel_hemisphere(ch)
        lobe = get_channel_lobe(ch)
        if hemi in ['left', 'right']:
            all_possible_lobes[hemi].add(lobe)

    for ch in unique_channels:
        lobes_by_hemisphere[get_channel_hemisphere(ch)].add(get_channel_lobe(ch))

    left_lobes = lobes_by_hemisphere.get('left', set())
    right_lobes = lobes_by_hemisphere.get('right', set())
    midline_lobes = lobes_by_hemisphere.get('midline', set())
    
    left_count = sum(1 for ch in unique_channels if get_channel_hemisphere(ch) == 'left')
    right_count = sum(1 for ch in unique_channels if get_channel_hemisphere(ch) == 'right')

    # Lobe sorting, in line with clinical habits (frontal-temporal-central-parietal-occipital)
    lobe_order = {'frontal': 0, 'temporal': 1, 'central': 2, 'parietal': 3, 'occipital': 4, 'unknown': 5}
    def format_lobe_str(lobes_set):
        if not lobes_set: 
            return ""
        
        sorted_lobes = sorted(list(lobes_set), key=lambda l: lobe_order.get(l, 99))
        # Special handling for singular/plural
        if len(sorted_lobes) == 1:
            return sorted_lobes[0] + " lobe"  # singular form
        else:
        # Using commas to join multiple lobes, and "and" for the last two
            if len(sorted_lobes) == 2:
                joined = " and ".join(sorted_lobes)
            else:
                joined = ", ".join(sorted_lobes[:-1]) + ", and " + sorted_lobes[-1]
            return joined + " lobes"  # plural form

    # 4. Generate description based on distribution pattern
    desc_parts = []
    
    # Pattern 1: Unilateral dominance or highly asymmetric
    # Judgment: One side has much more channels than the other (e.g., more than twice)
    if left_count > right_count * 2 or (left_count > 0 and right_count == 0):
    # Check if all lobes in the left hemisphere are covered
        if left_lobes == all_possible_lobes.get('left', set()):
            desc_parts.append("in the left hemisphere")
        else:
            desc_parts.append(f"in the {format_lobe_str(left_lobes)} of the left hemisphere")
    # If the right side is also slightly involved, add explanation
        if right_lobes:
            desc_parts.append(f" and the {format_lobe_str(right_lobes)} of the right hemisphere")
    
    elif right_count > left_count * 2 or (right_count > 0 and left_count == 0):
    # Check if all lobes in the right hemisphere are covered
        if right_lobes == all_possible_lobes.get('right', set()):
            desc_parts.append("in the right hemisphere")
        else:
            desc_parts.append(f"in the {format_lobe_str(right_lobes)} of the right hemisphere")
    # If the left side is also slightly involved, add explanation
        if left_lobes:
            desc_parts.append(f" and the {format_lobe_str(left_lobes)} of the left hemisphere")
    
    # Pattern 2: Bilateral distribution is relatively balanced
    # Judgment: Both sides are present, and the quantity difference is not large
    elif left_count > 0 and right_count > 0:
        all_lobes = left_lobes.union(right_lobes)
        desc_parts.append(f"in bilateral {format_lobe_str(all_lobes)}")

    # Pattern 3: Only midline
    elif not desc_parts and midline_lobes:
        desc_parts.append(f"in the {format_lobe_str(midline_lobes)} midline")

    # Add midline part
    if midline_lobes and (left_lobes or right_lobes):
         desc_parts.append(f" and {format_lobe_str(midline_lobes)} midline")

    return "".join(desc_parts) if desc_parts else "unknown region"

def group_events_by_time(events: list[dict], iou_threshold: float = 0.3) -> list[list[dict]]:
    """
    Group temporally overlapping events.
    For example, [{'time_range':[3,4]}, {'time_range':[3.1,3.9]}, {'time_range':[2,3]}]
    -> [[{'time_range': [2, 3]}], [{'time_range': [3, 4]}, {'time_range': [3.1, 3.9]}]]
    """
    if not events:
        return []
    
    events.sort(key=lambda x: x['time_range'][0])
    groups = []
    visited = [False] * len(events)
    
    for i in range(len(events)):
        if visited[i]:
            continue
        
        current_group = [events[i]]
        visited[i] = True
    # Merge the time range of the current event group
        group_time_range = list(events[i]['time_range'])
        for j in range(i + 1, len(events)):
            if not visited[j]:
                # Check if the new event overlaps with the merged time range of the current group
                if calculate_iou(events[j]['time_range'], group_time_range) > iou_threshold:
                    current_group.append(events[j])
                    visited[j] = True
                    # Update the merged time range
                    group_time_range[0] = min(group_time_range[0], events[j]['time_range'][0])
                    group_time_range[1] = max(group_time_range[1], events[j]['time_range'][1])
        groups.append(current_group)
    return groups

def prune_outliers_in_time_groups(groups: list[list[dict]], ch_list: list[str]) -> list[list[dict]]:
    """
    Perform spatial pruning within time groups.
    Pruning rule: When the number of channels on one side is at least 3 times that of the other side, and the minority side has no more than 2 channels, remove the minority side.
    Args:
        groups (list[list[dict]]): Output of `group_events_by_time`. list[dict] is a group, they overlap in time. dict is a bounding box.
        ch_list (list[str]): Channel names
    """
    pruned_groups = []
    for group in groups:
        # For very focal event groups (<=3 channels), we assume they are real and do not prune
        if len(group) <= 3:
            pruned_groups.append(group)
            continue

        hemisphere_counts = defaultdict(list)
        for event in group:
            hemi = get_channel_hemisphere(ch_list[event['channel_idx']])
            hemisphere_counts[hemi].append(event)
        
        left_count = len(hemisphere_counts.get('left', []))
        right_count = len(hemisphere_counts.get('right', []))
        
        new_group = list(group) # Copy for modification
        if left_count >= right_count * 3 and right_count <= 2:
            outliers = hemisphere_counts.get('right', [])
            new_group = [e for e in new_group if e not in outliers]
        elif right_count >= left_count * 3 and left_count <= 2:
            outliers = hemisphere_counts.get('left', [])
            new_group = [e for e in new_group if e not in outliers]

        if new_group: # Ensure the group is not empty after pruning
            pruned_groups.append(new_group)
    return pruned_groups

def prune_contained_events(events: list[dict]) -> list[dict]:
    """
    Prune events that are completely contained by others.
    """
    if not events:
        return []
    
    to_remove_indices = set()
    # Sort by duration in descending order, prefer to keep events with longer time ranges
    sorted_events = sorted(events, key=lambda e: e['time_range'][1] - e['time_range'][0], reverse=True)

    for i in range(len(sorted_events)):
        if i in to_remove_indices:
            continue
        container = sorted_events[i]
        container_len = container['time_range'][1] - container['time_range'][0]
        
        for j in range(i + 1, len(sorted_events)):
            if j in to_remove_indices:
                continue
            contained = sorted_events[j]

            # Check if it is completely contained in the same channel and time, container_len/8 allows container to expand a bit
            # For example, hfnoise is [0.1,9.9], muscle is [0.0,0.9], this design avoids treating these two as separate detections
            if (container['channel_idx'] == contained['channel_idx'] and
                container['time_range'][0]-container_len/8 <= contained['time_range'][0] and
                container['time_range'][1]+container_len/8 >= contained['time_range'][1]):
                to_remove_indices.add(j)
    pruned_events = [event for idx, event in enumerate(sorted_events) if idx not in to_remove_indices]
    return pruned_events

def time_range_to_epoch_indices(time_range: list[float], total_duration: int) -> tuple[int, int]:
    """
    Map a time range [0, 10] seconds directly to epoch indices.
    An epoch i corresponds to [i, i+1) seconds.
    """
    start_epoch = max(0, int(time_range[0]))
    end_epoch = min(total_duration, int(math.ceil(time_range[1])))
    return start_epoch, end_epoch

SYMMETRIC_PAIRS = {'FP1': 'FP2', 'F3': 'F4', 'C3': 'C4', 'P3': 'P4', 'O1': 'O2', 'F7': 'F8', 'T3': 'T4', 'T5': 'T6'}
def analyze_symmetry(
    freq_matrix: np.ndarray, 
    amp_matrix: np.ndarray, 
    mask: np.ndarray, 
    ch_map: dict[str, int]
) -> str:
    """
    Analyze amplitude and frequency symmetry between hemispheres.
    Logic:
    - Global conditions
        - At least min_comparable_epochs symmetric points required
        - The proportion of sustained asymmetry on one side must exceed sustained_threshold
    - Amplitude: The maximum value at symmetric points exceeds amp_min and (absolute difference exceeds amp_diff_abs or exceeds amp_diff_rel times)
    - Frequency: Ignore points above freq_cutoff, absolute difference not less than freq_diff_abs, minimum amplitude at symmetric points must exceed freq_amp_min
    """
    sustained_threshold = 0.5 # Only report if asymmetry duration exceeds sustained_threshold
    amp_diff_abs = 20.0       # Absolute amplitude difference threshold (uV)
    amp_diff_rel = 1.5        # Relative amplitude difference threshold (times)
    amp_min = 20.0            # Maximum amplitude on both sides must exceed this to consider asymmetry
    freq_diff_abs = 4         # Frequency difference threshold (Hz)
    freq_cutoff = 30          # Upper limit for frequency comparison (Hz)
    freq_amp_min = 10.0       # Minimum amplitude at symmetric points must exceed this to consider frequency asymmetry
    min_comparable_epochs = 3 # Minimum valid data points required for judgment

    num_epochs = freq_matrix.shape[1]
    
    asymmetry_details = defaultdict(lambda: {
        'amp_L_gt_R': 0, 'amp_R_gt_L': 0,
        'freq_L_gt_R': 0, 'freq_R_gt_L': 0,
        'total_comparable': 0
    })

    for left, right in SYMMETRIC_PAIRS.items():
        if left not in ch_map or right not in ch_map:
            continue

        l_idx, r_idx = ch_map[left], ch_map[right]
        pair_key = f"{left}/{right}"

        for epoch in range(num_epochs):
            if not (mask[l_idx, epoch] and mask[r_idx, epoch]):
                continue
            
            asymmetry_details[pair_key]['total_comparable'] += 1
            
            l_amp, r_amp = amp_matrix[l_idx, epoch], amp_matrix[r_idx, epoch]
            l_freq, r_freq = freq_matrix[l_idx, epoch], freq_matrix[r_idx, epoch]

            # 1. Amplitude asymmetry direction judgment
            is_asym_amp = ((abs(l_amp - r_amp) > amp_diff_abs) or \
                          (r_amp > 1e-6 and l_amp / r_amp > amp_diff_rel) or \
                          (l_amp > 1e-6 and r_amp / l_amp > amp_diff_rel)) and \
                          (max(l_amp,r_amp) > amp_min)
            
            if is_asym_amp:
                if l_amp > r_amp:
                    asymmetry_details[pair_key]['amp_L_gt_R'] += 1
                else:
                    asymmetry_details[pair_key]['amp_R_gt_L'] += 1

            # 2. Frequency asymmetry direction judgment (add high frequency filtering)
            # Only compare when both channel frequencies are below or equal to cutoff
            if l_freq <= freq_cutoff and r_freq <= freq_cutoff and min(l_freq,r_freq) > freq_amp_min:
                if abs(l_freq - r_freq) >= freq_diff_abs:
                    if l_freq > r_freq:
                        asymmetry_details[pair_key]['freq_L_gt_R'] += 1
                    else:
                        asymmetry_details[pair_key]['freq_R_gt_L'] += 1
    
    report_groups = defaultdict(list)
    
    for pair_key, counts in asymmetry_details.items():
        total = counts['total_comparable']
        
    # Ensure enough data points to support conclusion
        if total < min_comparable_epochs:
            continue

    # Check amplitude asymmetry
        if (counts['amp_L_gt_R'] / total) > sustained_threshold:
            report_groups['amp_L_gt_R'].append(pair_key)
        elif (counts['amp_R_gt_L'] / total) > sustained_threshold:
            report_groups['amp_R_gt_L'].append(pair_key)
            
    # Check frequency asymmetry
        if (counts['freq_L_gt_R'] / total) > sustained_threshold:
            report_groups['freq_L_gt_R'].append(pair_key)
        elif (counts['freq_R_gt_L'] / total) > sustained_threshold:
            report_groups['freq_R_gt_L'].append(pair_key)

    if not report_groups:
        return "" 

    desc_parts = []
    if report_groups['amp_L_gt_R']:
        pairs_str = ', '.join(report_groups['amp_L_gt_R'])
        desc_parts.append(f"on {pairs_str} leads, the amplitude on the left side is consistently higher than that on the right side")
    if report_groups['amp_R_gt_L']:
        pairs_str = ', '.join(report_groups['amp_R_gt_L'])
        desc_parts.append(f"on {pairs_str} leads, amplitude on the right side is consistently higher than that on the left side")
        
    if report_groups['freq_L_gt_R']:
        pairs_str = ', '.join(report_groups['freq_L_gt_R'])
        desc_parts.append(f"on {pairs_str} leads, left-sided frequency is consistently higher than right-sided")
    if report_groups['freq_R_gt_L']:
        pairs_str = ', '.join(report_groups['freq_R_gt_L'])
        desc_parts.append(f"on {pairs_str} leads, left-sided frequency is consistently higher than right-sided")

    if not desc_parts:
        return ""
    return "Interhemispheric asymmetry is observed in the background activity, manifested as:" + ";".join(desc_parts) + "."

def segment_time_series(
    values: np.ndarray, 
    threshold: float, 
    min_duration: int
) -> list[tuple[int, int, float]]:
    """
    Robust segmentation of time series.
    Improved segmentation logic:
    1. Initial segmentation: Split when the current value differs from the current segment average (> threshold).
    2. Iterative merging: Repeatedly merge adjacent segments that are similar in value (<= threshold) or too short (< min_duration), until no new merges occur in a pass.
       This ensures that segments with similar values (e.g., 12Hz and 13Hz) are merged even if initially split.
    3. Final filtering: Remove all segments whose final duration is still less than min_duration.
    """
    if np.all(np.isnan(values)):
        return []

    # --- 1. Initial segmentation: based on real-time average and threshold ---
    # This step only "splits", does not "merge"
    initial_segments = []
    start_idx = 0
    while start_idx < len(values):
        if np.isnan(values[start_idx]): # Skip leading NaN
            start_idx += 1
            continue

        end_idx = start_idx
        for i in range(start_idx + 1, len(values)):
            segment_so_far = values[start_idx : i]
            current_val = values[i]
            
            current_avg = np.nanmean(segment_so_far)
            
            # Only compare if both current value and current segment average are not NaN
            if not np.isnan(current_val) and not np.isnan(current_avg) and \
               abs(current_val - current_avg) > threshold:
                break # Difference too large, current segment ends at i-1
            end_idx = i # Otherwise, include current value in segment
        
        avg_val = np.nanmean(values[start_idx : end_idx + 1])
        initial_segments.append([start_idx, end_idx, avg_val])
        start_idx = end_idx + 1

    # --- 2. Iterative merging: merge similar or too short adjacent segments ---
    # Use a while loop until no merges occur in a full pass
    current_segments = [list(s) for s in initial_segments] # Convert to mutable list

    merged_in_pass = True
    while merged_in_pass and len(current_segments) > 1: # Continue looping as long as merges occur; stop if only one or none left
        merged_in_pass = False
        next_pass_segments = []
        
        # Process the first segment
        if current_segments:
            current_seg_to_process = current_segments[0]
        else:
            break # No segments left

        i = 1 # Start from the second segment
        while i < len(current_segments):
            next_seg_candidate = current_segments[i]
            
            # Calculate duration of current segment and next candidate
            current_dur = current_seg_to_process[1] - current_seg_to_process[0] + 1
            
          # Check merge conditions:
          # Condition 1: If their values are similar (within threshold)
          # Condition 2: Or, if current segment is too short and there are more segments to merge (avoid isolated short segments)
            can_merge_by_similarity = False
            if not np.isnan(current_seg_to_process[2]) and not np.isnan(next_seg_candidate[2]) and \
                abs(current_seg_to_process[2] - next_seg_candidate[2]) <= threshold:
                can_merge_by_similarity = True

            can_force_merge_if_too_short = False
          # Only consider forced merge if current segment is too short and not the only segment
            if current_dur < min_duration and len(current_segments) > 1:
                can_force_merge_if_too_short = True

            if can_merge_by_similarity or can_force_merge_if_too_short:
                # Perform merge: merge next_seg_candidate into current_seg_to_process
                current_seg_to_process[1] = next_seg_candidate[1] # Extend end time
                # Recalculate average value of merged segment using original `values` array
                current_seg_to_process[2] = np.nanmean(values[current_seg_to_process[0] : current_seg_to_process[1] + 1])
                merged_in_pass = True # Mark that a merge occurred in this pass
                i += 1 # Skip the next segment that was merged
            else:
                # Cannot merge: add current segment to next round list, then process next candidate
                next_pass_segments.append(current_seg_to_process)
                current_seg_to_process = next_seg_candidate
                i += 1
        
    # Add the last (or only) processed segment to the list
        next_pass_segments.append(current_seg_to_process)
        current_segments = next_pass_segments # Update, ready for next round of iterative merging

    # --- 3. Final filtering: remove all segments that still do not reach minimum duration ---
    # This step is necessary because even after merging, some (e.g., very short and very dissimilar to all neighbors) segments may still exist.
    final_filtered_segments = [
        s for s in current_segments 
        if (s[1] - s[0] + 1) >= min_duration
    ]
    
    return [tuple(s) for s in final_filtered_segments]

def analyze_temporal_evolution(
    freq_matrix: np.ndarray, 
    amp_matrix: np.ndarray, 
    mask: np.ndarray
) -> str:
    """
    Analyze the temporal evolution of background activity.
    - Both amplitude and frequency mask out high-frequency (>30Hz) data
    - Use robust segmentation algorithm and ensure minimum segment length
    - Even if only one parameter (frequency or amplitude) evolves, provide description for the other stable parameter
    - If no temporal evolution, provide basic background description
    """
    min_segment_duration = 3  # Minimum segment duration is 3 seconds
    freq_cutoff = 30          # Frequency analysis upper limit

    # --- 1. Data preparation ---
    ch_num = freq_matrix.shape[0]
    freq_clean = freq_matrix.copy().astype(float)
    amp_clean = amp_matrix.copy().astype(float)
    # Set mask part to nan
    freq_clean[~mask] = np.nan
    amp_clean[~mask] = np.nan
    # Set part above 30Hz to nan
    freq_high_mask = freq_clean > freq_cutoff
    freq_clean[freq_high_mask] = np.nan
    amp_clean[freq_high_mask] = np.nan
    # If the number of valid values in each column is less than 0.2 times the number of channels, set that column to nan
    non_nan_counts = np.sum(~np.isnan(freq_clean), axis=0)
    freq_clean[:, non_nan_counts < 0.2*ch_num] = np.nan
    amp_clean[:, non_nan_counts < 0.2*ch_num] = np.nan

    # Calculate the average value at each time point
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        epoch_avg_freq = np.nanmean(freq_clean, axis=0) # (10,)
        epoch_avg_amp = np.nanmean(amp_clean, axis=0)

    if np.count_nonzero(~np.isnan(epoch_avg_freq)) < min_segment_duration:
        return ""

    # --- 2. Segment the time series of frequency and amplitude ---
    freq_segments = segment_time_series(epoch_avg_freq, threshold=3.0, min_duration=min_segment_duration)
    amp_segments = segment_time_series(epoch_avg_amp, threshold=20.0, min_duration=min_segment_duration)

    # --- 3. Generate description text ---
    if not freq_segments or not amp_segments:
        return ""

    has_freq_evo = len(freq_segments) > 1
    has_amp_evo = len(amp_segments) > 1

    ############# Start of new and improved judgment logic #############
    # Special case: when amplitude is stable and below 15μV, small frequency fluctuations are considered clinically insignificant,
    # even if the algorithm detects frequency segmentation, it is regarded as stable and frequency evolution is not reported.
    if not has_amp_evo and has_freq_evo and amp_segments[0][2] < 15:
    # 1. Calculate the weighted average of all original frequency segments
        total_weighted_freq = 0
        total_duration = 0
        for start, end, avg_freq in freq_segments:
            duration = end - start + 1
            total_weighted_freq += avg_freq * duration
            total_duration += duration
        
    # 2. Create a new frequency segment representing the overall stable state
        if total_duration > 0:
            weighted_avg_freq = total_weighted_freq / total_duration
            # Use this new single segment to replace the original multiple segments
            overall_start = freq_segments[0][0]
            overall_end = freq_segments[-1][1]
            freq_segments = [(overall_start, overall_end, weighted_avg_freq)]
        
    # 3. Force the frequency evolution judgment to False
        has_freq_evo = False 
    ############# End of new and improved judgment logic #############

    # Case 1: Background is stable, no evolution
    if not has_freq_evo and not has_amp_evo:
        overall_avg_freq = freq_segments[0][2]
        overall_avg_amp = amp_segments[0][2]
        return (f"The background activity is stable, "
        f"predominantly composed of {get_amp_category(overall_avg_amp)}-amplitude "
        f"{get_freq_band_name(overall_avg_freq)} band "
        f"(at approximately {overall_avg_freq:.1f} Hz, amplitude about {overall_avg_amp:.1f} μV).")

    # Case 2: Background evolves
    descs = ["The background activity evolves over time: "]
    evo_parts = []
    
    # Describe amplitude part (evolution or stability)
    if has_amp_evo:
        parts = [f"{get_amp_category(avg_amp)} (~{avg_amp:.1f} μV) during {start}-{end+1}s" 
                for start, end, avg_amp in amp_segments]
        evo_parts.append("the amplitude is " + ", ".join(parts))
    else: # Amplitude is stable
        overall_avg_amp = amp_segments[0][2]
        evo_parts.append(f"Amplitude remained stable at {get_amp_category(overall_avg_amp)} (~{overall_avg_amp:.1f} μV)")

    # Describe frequency part (evolution or stability)
    if has_freq_evo:
        parts = [f"approximately {avg_freq:.1f} Hz ({get_freq_band_name(avg_freq)} band) during {start}-{end+1}s" 
                for start, end, avg_freq in freq_segments]
        evo_parts.append("the frequency is " + ", ".join(parts))
    else: # Frequency is stable
        overall_avg_freq = freq_segments[0][2]
        evo_parts.append(f"Frequency remained stable around {overall_avg_freq:.1f} Hz ({get_freq_band_name(overall_avg_freq)} band)")
        
    # Combine description and ensure only one period at the end
    final_desc = "".join(descs) + "; ".join(evo_parts) + "."
    return final_desc


########################
# Core Helper Function #
########################
def analysis_channel(channel_list:list[str]) -> str:
    '''
    Generate channel description text. Includes two conclusions:
    1. Whether it is the standard 10-20 system
    2. Reference method
    '''
    ret = ""
    STD_19_CHANNELS = {'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'}
    REFERENCE_MAP = {
        'AV': 'average reference',  # average reference
        'A1': 'left ear reference',  # left ear reference
        'A2': 'right ear reference',  # right ear reference
        'Aav': 'linked ears reference',  # linked ears reference
        'M1': 'left mastoid reference', # left mastoid reference
        'M2': 'right mastoid reference' # right mastoid reference
    }
    found_base_channels = set()
    found_references = set()

    # Traverse the list to collect all base channels and reference types
    for ch_name in channel_list:
        parts = ch_name.split('-')
        base_channel = parts[0]
        found_base_channels.add(base_channel)
        if len(parts) > 1:
            found_references.add(parts[1])

    # Conclusion 1: About channel names
    # Calculate the number of standard channels found in the list
    standard_channels_in_list = found_base_channels.intersection(STD_19_CHANNELS)
    num_standard_channels = len(standard_channels_in_list)
    if num_standard_channels == len(STD_19_CHANNELS) and len(found_base_channels) == num_standard_channels:
        ret += f"The channel layout uses the standard 10-20 system, "
    elif num_standard_channels == len(STD_19_CHANNELS):
        ret += f"The channel layout includes all standard 10-20 system channels but also contains non-standard channels, "
    else:
        channel_instance = ', '.join(standard_channels_in_list)
        ret += f"The channel layout contains {channel_instance} channels in {len(STD_19_CHANNELS)} standard 10-20 system channels, "
    
    # Conclusion 2: About reference method
    if len(found_references) == 0:
        ret += "without specified reference method in channel names."
    elif len(found_references) == 1:
        ref_code = found_references.pop()
        ref_name = REFERENCE_MAP.get(ref_code, f"unknown reference ({ref_code})")
        ret += f"using {ref_name}."
    elif found_references <= {"A1", "A2", "Aav"}:
        ret += "using ear reference."
    elif found_references <= {"M1", "M2"}:
        ret += "using mastoid reference."
    else:
        ret += "using bipolar reference."
    return ret

def analysis_artifact(
    YOLO_bboxes: list[dict],
    artifact_bboxes: list[dict],
    channel_list: list[str]
) -> str:
    '''
    Generate artifact description text. Uses labels:
    - YOLO_bboxes: "eyem", "eyer+", "eyer-", "hfnoise"
    - artifact_bboxes: "eog_v", "eog_left", "eog_right", "global_bad", "severe_artifact", "muscle", "nan_inf", "respiration"
    
    Description logic:
    1. Overall quality assessment
    2. Describe in order: respiratory artifact, extreme values, low-voltage, low-correlation channels, and high-frequency noise (higher priority masks lower)
    3. Eye movement artifact: FP1 and FP2 blink artifact -> sharp wave artifact in other channels -> F7 and F8 gaze artifact
    '''
    # --- 1. Initialization and data organization ---
    num_channels = len(channel_list)
    total_duration = 10 # All input data is 10s
    description_parts = []
    
    # Organize all artifacts by channel and type for easy lookup
    # Structure: {channel_idx: {'artifact_type': [[start, end], ...], ...}, ...}
    artifacts_by_channel = defaultdict(lambda: defaultdict(list))
    all_artifacts = YOLO_bboxes + artifact_bboxes # Merge YOLO and artifact recognition results
    for artifact in all_artifacts: # artifact is like {'channel_idx':9,'wave_type':'alpha','confidence':1.0,'time_range':[0,10.0]}
        artifact_type = artifact['wave_type']
        artifacts_by_channel[artifact['channel_idx']][artifact_type].append(artifact['time_range'])

    # --- 2. Identify eye movement artifacts and their derived artifacts ---
    # Blinks are only judged on FP1 and FP2, gaze on F7 and F8. If a blink co-occurs with a spike, it is considered a dipole rather than a blink artifact
    # Find the index of relevant channels in channel_list, default to -1 if not found, meaning no need to judge eye movement artifact
    fp1_idx = next((i for i, name in enumerate(channel_list) if name == 'FP1'), -1)
    fp2_idx = next((i for i, name in enumerate(channel_list) if name == 'FP2'), -1)
    f7_idx = next((i for i, name in enumerate(channel_list) if name == 'F7'), -1)
    f8_idx = next((i for i, name in enumerate(channel_list) if name == 'F8'), -1)
    
    # Store times of eye movement events
    raw_blink_times = []
    left_turn_times = []
    right_turn_times = []

    # Blink policy: if either FP1 or FP2 has 'eyem' or 'eog_v', treat it as a blink
    for ch_idx in [fp1_idx, fp2_idx]:
        if ch_idx != -1:
            for art_type in ['eyem', 'eog_v']:
                raw_blink_times.extend(artifacts_by_channel[ch_idx].get(art_type, []))

    # Find sharp waves induced by blinks
    # Step 1: Get all spike wave time ranges.
    all_spike_ranges = merge_time_ranges(
        [b['time_range'] for b in YOLO_bboxes if b['wave_type'] == 'spike']
    )

    # Step 2: Filter raw blink events; keep only those that do NOT coincide with spikes as reportable artifacts.
    reportable_blink_times = []
    if raw_blink_times:
        for b_time in raw_blink_times:
            # Check if the current blink time overlaps with any spike time
            is_coincident_with_spike = any(calculate_iou(b_time, s_range) > 0.3 for s_range in all_spike_ranges)
            # If not overlapping with spike, it is a pure artifact and can be reported.
            if not is_coincident_with_spike:
                reportable_blink_times.append(b_time)

    # Step 3: Based on reportable blinks, find induced artifact sharp waves
    blink_induced_sharps = defaultdict(list)
    if reportable_blink_times:
        for ch_idx in range(num_channels):
            if ch_idx in [fp1_idx, fp2_idx]:
                continue
            sharp_times = artifacts_by_channel[ch_idx].get('sharp', [])
            for s_time in sharp_times:
                for b_time in reportable_blink_times: # Use filtered blink times
                    if calculate_iou(s_time, b_time) > 0.3:
                        blink_induced_sharps[ch_idx].append(s_time)
                        break # Optimization: once a sharp wave is associated, no need to check further

    # Eye movement
    if f7_idx != -1:
        left_turn_times.extend(artifacts_by_channel[f7_idx].get('eog_left', []))
        left_turn_times.extend(artifacts_by_channel[f7_idx].get('eyer+', []))
        right_turn_times.extend(artifacts_by_channel[f7_idx].get('eyer-', []))
    if f8_idx != -1:
        right_turn_times.extend(artifacts_by_channel[f8_idx].get('eog_right', []))
        right_turn_times.extend(artifacts_by_channel[f8_idx].get('eyer+', []))
        left_turn_times.extend(artifacts_by_channel[f8_idx].get('eyer-', []))

    # --- 3. Overall quality assessment ---
    quality_artifacts = {"global_bad", "severe_artifact", "muscle", "hfnoise", "nan_inf", "respiration"}
    artifact_mat = np.zeros((num_channels,total_duration),dtype=bool)
    for ch_idx, arts in artifacts_by_channel.items():
        for art_type, ranges in arts.items():
            if art_type in quality_artifacts: # Included in data quality assessment
                for start, end in ranges:
                    # Map time range to discrete indices of background
                    start_bin = int(np.floor(start))
                    end_bin = int(np.ceil(end))
                    for i in range(start_bin, min(end_bin, total_duration)):
                        artifact_mat[ch_idx, i] = True
    artifact_ratio = np.sum(artifact_mat) / (num_channels * total_duration)
    if artifact_ratio < 0.2:
        quality_desc = "Overall data quality is good."
    elif artifact_ratio < 0.5:
        quality_desc = "Data quality is fair with some artifact interference."
    else:
        quality_desc = "Data quality is poor and the signal is severely contaminated by artifacts."
    description_parts.append(quality_desc)

    # --- 4. Generate specific artifact descriptions ---
    # 1) Items are ordered by priority; earlier items mask later ones
    ARTIFACT_GROUPS = [
        (['respiration'], "respiratory artifacts"),
        (['nan_inf'], "extreme values"),
        (['flat'], "low-voltage channels"),
        (['global_bad', 'severe_artifact'], "low-correlation channels"),
        (['muscle', 'hfnoise'], "high-frequency noise")
    ]

    # Track spatiotemporal regions occupied by higher-priority artifacts
    # to avoid double-reporting lower-priority artifacts in those regions
    priority_artifact_regions = defaultdict(list) # {channel_idx: [[start1, end1], ...]}

    for (art_types, template) in ARTIFACT_GROUPS:
    # Step 1: Collect all raw artifact events for the current group.
        # We need this original list to add it completely to the masked region later.
        original_events_for_group = []
        for ch_idx, arts_by_type in artifacts_by_channel.items():
            for art_type, ranges in arts_by_type.items():
                if art_type in art_types:
                    for r in ranges:
                        original_events_for_group.append({
                            'channel_idx': ch_idx,
                            'wave_type': art_type,
                            'time_range': r
                        })
        
        if not original_events_for_group:
            continue

    # Step 2: Filter using accumulated masked regions to drop covered events.
        events_to_report = []
        for event in original_events_for_group:
            is_covered = False
            occupied_ranges = priority_artifact_regions.get(event['channel_idx'], [])
            for described_range in occupied_ranges:
                if calculate_iou(event['time_range'], described_range) > 0.1:
                    is_covered = True
                    break
            
            if not is_covered:
                events_to_report.append(event)
        
    # Step 3: Report only the filtered remaining events.
        if events_to_report:
            # This part of the logic is exactly the same as your original code, but operates on events_to_report
            pruned_events = prune_contained_events(events_to_report)
            time_based_groups = group_events_by_time(pruned_events, iou_threshold=0.35)
            
            families = defaultdict(list) 
            for group in time_based_groups:
                group_channels = [channel_list[event['channel_idx']] for event in group]
                group_time_ranges = [event['time_range'] for event in group]
                
                location_summary = summarize_channel_locations(group_channels, channel_list)
                families[location_summary].extend(group_time_ranges)
                
            for location_summary, all_time_ranges in families.items():
                time_desc = format_duration_description(all_time_ranges, total_duration, threshold=0.8)
                if time_desc == "throughout the recording period":
                    if 'channels' in template:
                        desc = f"{template} are present {location_summary} {time_desc}."
                    elif 'high-frequency noise' in template:
                        desc = f"{template} is persistently observed {location_summary} {time_desc}."
                    else:
                        desc = f"{template} are persistently observed {location_summary} {time_desc}."
                elif 'high-frequency noise' in template:
                    final_merged_ranges = merge_time_ranges(all_time_ranges)
                    time_str = format_time_ranges_string(final_merged_ranges)
                    desc = f"{template} is observed {location_summary} {time_str}."
                else:
                    final_merged_ranges = merge_time_ranges(all_time_ranges)
                    time_str = format_time_ranges_string(final_merged_ranges)
                    desc = f"{template} are observed {location_summary} {time_str}."             
                description_parts.append(desc)

    # Step 4: Record the raw event footprints into the mask map so lower-priority artifacts are masked in later loops.
    # Regardless of whether some events were reported, the group's full footprint is used to mask later groups.
        for event in original_events_for_group:
            ch_idx = event['channel_idx']
            time_range = event['time_range']
            regions = priority_artifact_regions[ch_idx]
            regions.append(time_range)
            priority_artifact_regions[ch_idx] = merge_time_ranges(regions)

    # 2) Then describe eye movement artifacts
    if reportable_blink_times:
        merged_blinks = merge_time_ranges(reportable_blink_times)
        time_str = format_time_ranges_string(merged_blinks)
        description_parts.append(f"Frontopolar leads (FP1, FP2) demonstrate eyeblink artifacts {time_str}.")
        if blink_induced_sharps:
            sharp_descs = []
            sharp_ranges = []
            for ch_idx, ranges in blink_induced_sharps.items():
                sharp_descs.append(f"{channel_list[ch_idx]}")
                sharp_ranges.extend(ranges)
            location_summary = summarize_channel_locations(sharp_descs, channel_list)
            merged_sharp_ranges = merge_time_ranges(sharp_ranges)
            time_str = format_time_ranges_string(merged_sharp_ranges)
            description_parts.append(f"Consequently, artifact-induced sharp waves are observed {location_summary} {time_str}.")

    if left_turn_times:
        merged_turns = merge_time_ranges(left_turn_times)
        time_str = format_time_ranges_string(merged_turns)
        description_parts.append(f"Left-gaze artifacts detected in temporal leads (F7, F8) {time_str}.")
        
    if right_turn_times:
        merged_turns = merge_time_ranges(right_turn_times)
        time_str = format_time_ranges_string(merged_turns)
        description_parts.append(f"Right-gaze artifacts detected in temporal leads (F7, F8) {time_str}.")
    # --- 5. Combine and return the final result ---
    return "".join(description_parts)


def analysis_discharge(
        YOLO_bboxes: list[dict],
        channel_list: list[str]
) -> str:
    '''
    Generate description text for epileptiform discharges and physiological sleep waves. Uses labels:
    - YOLO_bboxes: 'sharp', 'spike', 'spsw', 'spindle', 'Kcomplex', 'eyem'

    Logic:
    1. Analyze spindle and Kcomplex
    2. Filter out sharp waves generated by eyem
    3. Vertex sharp analysis: globally no blink, locally no discharge, location condition (frontal or parietal) and central sharp, no occipital sharp
    4. Epileptiform discharge analysis, left-right dipole analysis.
    '''
    # --- 0. If there are no events of interest, return directly ---
    interest_waves = {'sharp', 'spike', 'spsw', 'spindle', 'Kcomplex'}
    if not any(b['wave_type'] in interest_waves for b in YOLO_bboxes):
        return "No definite epileptiform discharges or physiological sleep waves are observed."
    report_parts = []
    
    # --- 1. Data classification ---
    sleep_events = defaultdict(list) # {'spindle':[{'channel_idx':2,'wave_type':'spindle','confidence': 0.9,'time_range':[1.2,2.0]},...],...}
    discharge_events = [] # [{'channel_idx': 12, 'wave_type': 'spike', 'confidence': 0.95, 'time_range': [3.5, 3.7]},...]
    eyem_events = [] # [{'channel_idx': 11, 'wave_type': 'eyem', 'confidence': 0.95, 'time_range': [3.5, 3.7]},...]
    for bbox in YOLO_bboxes:
        wave_type = bbox['wave_type']
        if wave_type in ['spindle', 'Kcomplex']:
            sleep_events[wave_type].append(bbox)
        elif wave_type in ['sharp', 'spike', 'spsw']:
            discharge_events.append(bbox)
        elif wave_type == 'eyem':
            eyem_events.append(bbox)

    # --- 2. Sleep wave analysis ---
    is_sleep_stage = False
    if sleep_events:
        is_sleep_stage = True
        report_parts.append("This EEG recording is obtained during sleep.")
        for wave_type, events in sleep_events.items():
            location_summary = summarize_channel_locations([channel_list[e['channel_idx']] for e in events], channel_list)
            time_str = format_time_ranges_string(merge_time_ranges([e['time_range'] for e in events]))
            wave_name = WAVE_TYPE_MAP.get(wave_type)
            report_parts.append(f"{wave_name} are present {location_summary} {time_str}.")

    # --- 3. Preprocessing discharge waves: filter out artifact sharp waves caused by blink artifact (eyem) ---
    eyem_fp_ranges = merge_time_ranges(
        [b['time_range'] for b in eyem_events if channel_list[b['channel_idx']].startswith(('FP1', 'FP2'))]
    )
    all_spike_ranges = merge_time_ranges( # If there is also spike, it should be spike-induced eyem
        [b['time_range'] for b in discharge_events if b['wave_type'] == 'spike']
    )
    valid_discharges = []
    for bbox in discharge_events:
        if bbox['wave_type'] == 'sharp':
            # Check if the current sharp wave overlaps in time with blink artifact
            is_coincident_with_blink = any(calculate_iou(bbox['time_range'], r) > 0.5 for r in eyem_fp_ranges)
            # By default, not considered artifact, only marked as artifact if specific conditions are met
            is_artifact = False
            if is_coincident_with_blink:
                # If overlapping with blink, further check if also overlapping with spike
                is_coincident_with_spike = any(calculate_iou(bbox['time_range'], r_spike) > 0.5 for r_spike in all_spike_ranges)
                # Core judgment: only when sharp wave overlaps with blink but does NOT overlap with any spike, it is considered artifact
                if not is_coincident_with_spike:
                    is_artifact = True
            # If not considered artifact, add to valid discharge list
            if not is_artifact:
                valid_discharges.append(bbox)
        else:
        # For spike and spike-slow wave (spsw), directly add to valid discharge list
            valid_discharges.append(bbox)
    
    if not valid_discharges: # If no discharge remains after filtering, return directly
        if not report_parts: return "No definite epileptiform discharges or physiological sleep waves are observed."
        else: return "".join(report_parts)

    # -- 4. Vertex sharp analysis --
    sharp_events = [b for b in valid_discharges if b['wave_type'] == 'sharp']
    vertex_sharps_indices = set() # Use id to mark processed events
    # Condition 1: Global check. If any eyem is detected on FP1/FP2, do not perform vertex sharp analysis.
    is_blink_present_globally = bool(eyem_fp_ranges)
    if sharp_events and not is_blink_present_globally:
        synchronous_sharps_groups = group_events_by_time(sharp_events, iou_threshold=0.3)
        
    # Prepare a list of spike and SPSW for subsequent instant check
        spike_spsw_events = [b for b in valid_discharges if b['wave_type'] in ['spike', 'spsw']]

        for group in synchronous_sharps_groups:
            # Condition 2: Check brain region distribution. At the same time (frontal or parietal) and central sharp, and no occipital sharp
            lobes = {get_channel_lobe(channel_list[b['channel_idx']]) for b in group}
            is_location_match = ('frontal' in lobes or 'parietal' in lobes) and \
                                'central' in lobes and \
                                'occipital' not in lobes
            if not is_location_match:
                continue

            # Condition 3: Instant check. Check if there is spike or SPSW at the same time as sharp wave
            is_spike_spsw_concurrent = False
            # Get the merged time range of the current sharp wave cluster
            group_time_range = merge_time_ranges([b['time_range'] for b in group])[0]
            for event in spike_spsw_events:
                # Any overlap in time (iou > 0) is considered "simultaneous occurrence"
                if calculate_iou(group_time_range, event['time_range']) > 0:
                    is_spike_spsw_concurrent = True
                    break
            
            # Final judgment: must meet location condition and not occur simultaneously with spike/SPSW
            if not is_spike_spsw_concurrent:
                # If all conditions are met, it is judged as vertex sharp
                channels = [channel_list[ch_idx] for ch_idx in sorted(list(set([b['channel_idx'] for b in group])))]
                time_ranges = merge_time_ranges([b['time_range'] for b in group])
                prefix = "vertex waves" if is_sleep_stage else "sharp waves resembling vertex waves"
                location_summary = summarize_channel_locations(channels, channel_list)
                report_parts.append(f"{prefix} are demonstratd {location_summary} {format_time_ranges_string(time_ranges)}.")
                # Mark these sharp waves as processed to avoid reporting as ordinary discharge later
                for event in group:
                    vertex_sharps_indices.add(id(event))

    # --- 5. Epileptiform discharge analysis ---
    # Previously filtered out artifact sharp waves caused by blink, now filter out vertex sharp
    remaining_discharges = [b for b in valid_discharges if id(b) not in vertex_sharps_indices]
    if not remaining_discharges: 
        if not report_parts: return "No definite epileptiform discharges or physiological sleep waves are observed."
        else: return "".join(report_parts)
    
    # Now only sharp, spike, and spsw confirmed as epileptiform discharge remain
    # The idea is to create families by spatial similarity, continuously adding waveforms, channels, and times
    # 5a. Initial grouping by temporal synchrony
    discharge_groups = group_events_by_time(remaining_discharges, iou_threshold=0.3)
    # Prune time groups before clustering
    pruned_discharge_groups = prune_outliers_in_time_groups(discharge_groups, channel_list)

    # 5b. Iterative merging to form final "event families"
    final_families = []
    JACCARD_THRESHOLD = 0.3 

    for group in pruned_discharge_groups:
        current_channels = {channel_list[e['channel_idx']] for e in group}
        current_wave_types = {WAVE_TYPE_MAP[e['wave_type']] for e in group}
        current_time_ranges = merge_time_ranges([e['time_range'] for e in group])

        best_match_idx = -1
        max_jaccard = -1.0

        for i, family in enumerate(final_families):            
            # Calculate spatial similarity
            intersection = len(family['channels'].intersection(current_channels))
            union = len(family['channels'].union(current_channels))
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                best_match_idx = i
        
        if max_jaccard > JACCARD_THRESHOLD:
            # Merge into the best matching family
            final_families[best_match_idx]['channels'].update(current_channels)
            final_families[best_match_idx]['wave_types'].update(current_wave_types)
            final_families[best_match_idx]['time_ranges_list'].append(current_time_ranges)
        else:
            # Create new family
            final_families.append({
                'channels': current_channels,
                'wave_types': current_wave_types,
                'time_ranges_list': [current_time_ranges]
            })

    # 5c. Analyze each final family and generate report
    for family in final_families:
    # [Modification] Use the length of family['time_ranges_list'] to determine the number of event clusters, which more accurately reflects the number of "time clusters" than len(group)
        num_time_clusters = len(family['time_ranges_list'])
        all_time_ranges = [r for sublist in family['time_ranges_list'] for r in sublist]
        
        location_summary = summarize_channel_locations(list(family['channels']), channel_list)
        wave_types_str = ' and '.join(sorted(list(family['wave_types'])))

    # The threshold for periodicity judgment can be adjusted according to clinical experience, here set as >3 time clusters
        if num_time_clusters > 3:
            sorted_ranges = sorted(all_time_ranges, key=lambda x: x[0])
            total_duration = sorted_ranges[-1][1] - sorted_ranges[0][0]
            freq = num_time_clusters / total_duration if total_duration > 1 else num_time_clusters
            desc = f"Periodic {wave_types_str} are observed {location_summary} with approximate frequency {freq:.1f} Hz"
        else:
            final_merged_ranges = merge_time_ranges(all_time_ranges)
            time_str = format_time_ranges_string(final_merged_ranges)
            desc = f"Sporadic {wave_types_str} are present {location_summary} {time_str}"

        # Dipole composed of eyem and spike with sharp
        is_potential_dipole = False
        # Feature 1: Event contains sharp and spike
        if 'sharps' in family['wave_types'] and 'spikes' in family['wave_types']:
            # Feature 2: Event time overlaps with blink event on FP1/FP2
            if any(calculate_iou(r, eyem_r) > 0.3 for r in all_time_ranges for eyem_r in eyem_fp_ranges):
                is_potential_dipole = True
        
        if is_potential_dipole:
            # When basic conditions for anterior-posterior dipole are met, further check for left-right dipole
            # 1. Find all blink events that overlap in time with current discharge event (not just FP1/FP2)
            coincident_eyem_events = []
            for eyem_event in eyem_events:
                if any(calculate_iou(eyem_event['time_range'], r) > 0.3 for r in all_time_ranges):
                    coincident_eyem_events.append(eyem_event)
            
            # 2. Analyze hemisphere distribution of these blink events on channels other than frontal pole
            left_side_eyem_count = 0
            right_side_eyem_count = 0
            for event in coincident_eyem_events:
                ch_name = channel_list[event['channel_idx']]
                if ch_name.startswith('F'):
                    continue # Exclude frontal pole leads
                
                hemisphere = get_channel_hemisphere(ch_name)
                if hemisphere == 'left':
                    left_side_eyem_count += 1
                elif hemisphere == 'right':
                    right_side_eyem_count += 1
            
            # 3. Decide dipole type based on distribution
            # Condition: If blink artifact shows obvious lateralization on other channels
            is_lateralized = (left_side_eyem_count > 0 and right_side_eyem_count == 0) or \
                             (right_side_eyem_count > 0 and left_side_eyem_count == 0)
            if min(left_side_eyem_count,right_side_eyem_count) > 0:
                if max(left_side_eyem_count,right_side_eyem_count) / min(left_side_eyem_count,right_side_eyem_count) > 2:
                    is_lateralized = True

            if is_lateralized:
                # If lateralized, report left-right dipole
                desc += ", suggesting a left-right dipole phenomenon"
            else:
                # Otherwise, fall back to original anterior-posterior dipole description
                desc += ", suggesting an anterior-posterior dipole phenomenon"
            
        desc += '.'
        report_parts.append(desc)

    # 5d. Summarize "spike-sharp" dipole phenomenon at the end of the report
    observed_dipole_patterns = [] # Used to store observed dipole pattern directions
    for group in pruned_discharge_groups:
        hemi_waves = defaultdict(lambda: {'spike': 0, 'sharp': 0})
        for event in group:
            channel_name = channel_list[event['channel_idx']]
            if channel_name in ['FP1', 'FP2']: # Trick: dipole does not consider FP1 and FP2, they are too easily similar
                continue
            hemi = get_channel_hemisphere(channel_name)
            if hemi in ['left', 'right'] and event['wave_type'] in ['spike', 'sharp']:
                hemi_waves[hemi][event['wave_type']] += 1
        
        left_is_pure_spike = hemi_waves['left']['spike'] > 0 and hemi_waves['left']['sharp'] == 0
        right_is_pure_sharp = hemi_waves['right']['sharp'] > 0 and hemi_waves['right']['spike'] == 0
        
        left_is_pure_sharp = hemi_waves['left']['sharp'] > 0 and hemi_waves['left']['spike'] == 0
        right_is_pure_spike = hemi_waves['right']['spike'] > 0 and hemi_waves['right']['sharp'] == 0
        
        if (left_is_pure_spike and right_is_pure_sharp):
            observed_dipole_patterns.append("L_spike_R_sharp")
        elif (right_is_pure_spike and left_is_pure_sharp):
            observed_dipole_patterns.append("R_spike_L_sharp")
            
        # If dipole pattern is found, perform consistency check
    if len(observed_dipole_patterns) > 1: # At least observed twice
        # Check if all observed patterns are the same
        unique_patterns = set(observed_dipole_patterns)
        
        # Only generate specific description if the pattern is unique and persistent (i.e., only one pattern)
        if len(unique_patterns) == 1:
            pattern = unique_patterns.pop() # Get the unique pattern
            if pattern == "L_spike_R_sharp":
                spike_side = "left hemisphere"
                sharp_side = "right hemisphere"
            else: # "R_spike_L_sharp"
                spike_side = "right hemisphere"
                sharp_side = "left hemisphere"
            dipole_desc = f"Additionally, a consistent dipole pattern is observed with spike waves in the {spike_side} and sharp waves in the contralateral {sharp_side}."
            report_parts.append(dipole_desc)
        # If there are conflicting patterns (len(unique_patterns) > 1), do not add any summary description to avoid confusion.

    if not report_parts:
        return "No definite epileptiform discharges or physiological sleep waves are observed."
    return "".join(report_parts)


def analysis_background(
        YOLO_bboxes:list[dict],
        artifact_bboxes:list[dict],
        dominant_freq_matrix:np.ndarray,
        max_amp_matrix:np.ndarray,
        channel_list:list[str]
) -> str:
    '''
    Generate background description text. Logic:
    1. Mask all non-background events
    2. Describe posterior alpha waves (frequency, amplitude, and length requirements, allowing brief discontinuities)
    3. Describe delta waves, widespread and persistent
    4. Temporal evolution analysis, symmetry analysis
    '''
    description_parts = []
    num_channels, total_duration = dominant_freq_matrix.shape
    channel_to_idx = {name: i for i, name in enumerate(channel_list)}

    # --- 0. Create background mask, mask all non-background events ---
    background_mask = np.ones_like(dominant_freq_matrix, dtype=bool)
    yolo_events_to_mask = [b for b in YOLO_bboxes if b['wave_type'] not in ['alpha', 'delta']] # Mask all except alpha and delta events
    events_to_mask = artifact_bboxes + yolo_events_to_mask

    for bbox in events_to_mask:
        ch_idx = bbox['channel_idx']
        if 0 <= ch_idx < num_channels:
            start_epoch, end_epoch = time_range_to_epoch_indices(bbox['time_range'], total_duration)
            background_mask[ch_idx, start_epoch:end_epoch] = False

    # --- 1. Analyze posterior dominant rhythm (PDR - Alpha), only on unmasked background ---
    # Alpha rhythm mask: frequency in [9, 14) Hz and amplitude > 20 uV
    alpha_mask = (dominant_freq_matrix >= 9) & (dominant_freq_matrix < 14) & (max_amp_matrix > 20) & background_mask
    pdr_channel_names = {'O1', 'O2'} # Mainly focus on O1, O2
    pdr_indices = [idx for name, idx in channel_to_idx.items() if name in pdr_channel_names]

    pdr_ranges = []
    
    if pdr_indices:
        pdr_mask_any_channel = alpha_mask[pdr_indices, :].any(axis=0)
    # Fill single "holes" to smooth Alpha rhythm detection. Change False between two Trues to True
        for i in range(1, len(pdr_mask_any_channel) - 1):
            if pdr_mask_any_channel[i-1] and not pdr_mask_any_channel[i] and pdr_mask_any_channel[i+1]:
                pdr_mask_any_channel[i] = True

        continuity_req = 2 # Require alpha wave to be at least continuity_req long
        continuous_alpha_starts = np.convolve(pdr_mask_any_channel, np.ones(continuity_req), mode='valid') >= continuity_req # (10-continuity_req,)
        true_start_indices = np.where(continuous_alpha_starts)[0] # Indices that meet the condition, from which continuity_req are all True
        if true_start_indices.size > 0:
            # Find consecutive index blocks (e.g., [0,1,2, 5,6] -> [[0,1,2], [5,6]])
            blocks = np.split(true_start_indices, np.where(np.diff(true_start_indices) != 1)[0] + 1)
            for block in blocks:
                if block.size > 0:
                    start_sec = block[0]
                    end_sec = block[-1] + continuity_req - 1
                    pdr_ranges.append([float(start_sec), float(end_sec + 1)])

    # If PDR is found, generate description
    if pdr_ranges:
        pdr_freqs = []
        pdr_amps = []
        merged_ranges = merge_time_ranges(pdr_ranges)
    # Collect frequency and amplitude during detected time periods
        for start, end in merged_ranges:
            start_epoch, end_epoch = time_range_to_epoch_indices([start, end], total_duration)
            for ch_idx in pdr_indices:
                for epoch in range(start_epoch, end_epoch):
                    if alpha_mask[ch_idx, epoch]:
                        pdr_freqs.append(dominant_freq_matrix[ch_idx, epoch])
                        pdr_amps.append(max_amp_matrix[ch_idx, epoch])

        if pdr_freqs:
            avg_freq = np.mean(pdr_freqs)
            avg_amp = np.mean(pdr_amps)
            time_desc = format_duration_description(merged_ranges, total_duration, threshold=1.0)
            pdr_desc = f"{time_desc}, the posterior head region demonstrates alpha rhythm at approximately {avg_freq:.1f} Hz, with {get_amp_category(avg_amp)} amplitude (~{avg_amp:.1f} μV), suggesting the subject is likely in an awake, eyes-closed state."
            description_parts.append(pdr_desc)

    # --- 2. Describe Delta activity ---
    # Delta rhythm mask: frequency in (0, 4) Hz and amplitude > 30 uV
    delta_mask = (dominant_freq_matrix < 4) & (max_amp_matrix > 30) & background_mask
    delta_events = []

    # 2a. Identify widespread Delta (synchronous in more than half of channels)
    widespread_epochs = np.where(np.sum(delta_mask, axis=0) > num_channels / 2)[0]
    for epoch in widespread_epochs:
        involved_channels_indices = np.where(delta_mask[:, epoch])[0]
        delta_events.append({
            'channel_indices': list(involved_channels_indices),
            'time_range': [float(epoch), float(epoch + 1)]
        })

    # 2b. Identify persistent Delta (single channel continuous for more than 3 seconds)
    for ch_idx in range(num_channels):
        continuous_delta = np.convolve(delta_mask[ch_idx, :], np.ones(3), mode='valid') >= 3
        is_in_range = False
        for i, is_continuous in enumerate(continuous_delta):
            if is_continuous and not is_in_range:
                start_time = i
                is_in_range = True
            elif not is_continuous and is_in_range and start_time != -1:
                end_time = i + 2
                freqs = dominant_freq_matrix[ch_idx, start_time:end_time+1][delta_mask[ch_idx, start_time:end_time+1]].tolist()
                amps = max_amp_matrix[ch_idx, start_time:end_time+1][delta_mask[ch_idx, start_time:end_time+1]].tolist()
                delta_events.append({
                    'channel_indices': [ch_idx],
                    'time_range': [float(start_time), float(end_time + 1)],
                    'freqs': freqs, 'amps': amps
                })
                is_in_range = False
                start_time = -1
        if is_in_range and start_time != -1:
            freqs = dominant_freq_matrix[ch_idx, start_time:total_duration][delta_mask[ch_idx, start_time:total_duration]].tolist()
            amps = max_amp_matrix[ch_idx, start_time:total_duration][delta_mask[ch_idx, start_time:total_duration]].tolist()
            delta_events.append({
                'channel_indices': [ch_idx],
                'time_range': [float(start_time), float(total_duration)],
                'freqs': freqs, 'amps': amps
            })

    # 2c. Merge and describe all Delta events
    if delta_events:
        all_delta_channels = set()
        all_delta_ranges = []
        all_delta_freqs = []
        all_delta_amps = []
        
        for event in delta_events:
            all_delta_channels.update(event['channel_indices'])
            all_delta_ranges.append(event['time_range'])
            all_delta_freqs.extend(event.get('freqs', []))
            all_delta_amps.extend(event.get('amps', []))
        
        location_summary = summarize_channel_locations([channel_list[idx] for idx in sorted(list(all_delta_channels))], channel_list)
        time_desc = format_duration_description(all_delta_ranges, total_duration)
        
        if all_delta_freqs and all_delta_amps:
            avg_freq = np.mean(all_delta_freqs)
            avg_amp = np.mean(all_delta_amps)
            delta_desc = (
                            f"delta activity is observed {location_summary} {time_desc}, "
                            f"with frequency approximately {avg_freq:.1f} Hz and {get_amp_category(avg_amp)} amplitude (~{avg_amp:.1f} μV)."
                        )
        else:
            delta_desc = f"delta activity is observed {location_summary} {time_desc}."
        if not delta_desc.startswith('in the whole brain throughout the recording period demonstrates delta activity'): # Trick, this will be described in temporal evolution analysis, so no need to describe again here
            description_parts.append(delta_desc)

    # --- 3. Analyze remaining background ---
    if np.sum(background_mask) > background_mask.size * 0.5: # Ensure enough background remains
        # 3a. Temporal evolution analysis
        evo_desc = analyze_temporal_evolution(dominant_freq_matrix, max_amp_matrix, background_mask)
        if evo_desc: description_parts.append(evo_desc)
        # 3b. Symmetry analysis
        sym_desc = analyze_symmetry(dominant_freq_matrix, max_amp_matrix, background_mask, channel_to_idx)
        if sym_desc: description_parts.append(sym_desc)
        
    return "".join(description_parts)

def normalize_description(description: str) -> str:
    """
    Normalize description text to ensure compliance with English grammar rules:
    1. Ensure there is a space after each period
    2. Ensure the first letter after a period is capitalized
    """
    if not description:
        return ""
    
    # Step 1: Ensure there is a space after every period (unless already followed by a space)
    description = re.sub(r'\.(?!\s|$)', '. ', description)
    description = re.sub(r':(?!\s|$)', ': ', description)

    # Step 2: Fix spaces incorrectly added in decimals
    # Check if both sides of the period are digits, if so, remove the space
    chars = list(description)
    new_chars = []
    i = 0
    n = len(chars)
    
    while i < n:
        if chars[i] == '.' and i > 0 and i < n-1:
            # Check if both sides of the period are digits
            prev_is_digit = chars[i-1].isdigit()
            next_is_digit = chars[i+2].isdigit() if i+2 < n else False
            
            if prev_is_digit and next_is_digit:
                # If it is a decimal point, remove the space
                # But keep the period itself
                new_chars.append('.')
                
                # Skip the added space
                if i+1 < n and chars[i+1] == ' ':
                    i += 1  
            else:
                # Not a decimal, keep the period and possible space
                new_chars.append('.')
                if i+1 < n and chars[i+1] == ' ':
                    new_chars.append(' ')
                    i += 1  
        else:
            new_chars.append(chars[i])
        i += 1
    
    description = ''.join(new_chars)

    # Step 2: Capitalize the first letter of each sentence
    sentences = []
    for sentence in description.split('. '):
        sentence = sentence.strip()
        if sentence:
            # Ensure the first letter of the sentence is capitalized
            if sentence and not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]  
            sentences.append(sentence)
    
    # Step 3: Rejoin sentences
    normalized = '. '.join(sentences)
    
    return normalized

############
# Core API #
############
def text_generator(
        YOLO_bboxes:list[dict],
        artifact_bboxes:list[dict],
        dominant_freq_matrix:np.ndarray,
        max_amp_matrix:np.ndarray,
        channel_list:list[str]
        ) -> str:
    '''
    Rule-based text description (caption) generation. English version.

    Args:
        YOLO_bboxes (list[dict]): YOLO waveform recognition output;
            Dict like {'channel_idx':9,'wave_type':'alpha','confidence':1.0,'time_range':[0,10.0]};
            Waveforms: ['sharp','spike','spsw','alpha','delta','spindle','Kcomplex','eyem','eyer+','eyer-','hfnoise']
        artifact_bboxes (list[dict]): Artifact recognition output (dict same as YOLO_bboxes);
            Waveforms: ["global_bad","severe_artifact","muscle","eog_v","eog_left","eog_right","flat","nan_inf","respiration"]
        dominant_freq_matrix (ndarray[C,10]): Background recognition - dominant frequency
        max_amp_matrix (ndarray[C,10]): Background recognition - amplitude
        channel_list (list[str]): Channel name order. Like ['FP1-AV', 'FP2-AV', 'F3-AV', 'F4-AV', 'C3-AV', 'C4-AV',...]

    Returns:
        str: Rule-based text description
    '''
    description = ''
    
    # First step: look at reference method
    description += analysis_channel(channel_list)

    # For background analysis, we do not care about reference method
    channel_list_noref = [ch.split('-')[0] for ch in channel_list]

    # Second step: look at artifacts
    description += analysis_artifact(YOLO_bboxes, artifact_bboxes, channel_list_noref)

    # Third step: look at discharges
    description += analysis_discharge(YOLO_bboxes, channel_list_noref)

    # Fourth step: look at background rhythms
    description += analysis_background(YOLO_bboxes, artifact_bboxes, dominant_freq_matrix, max_amp_matrix, channel_list_noref)

    # Final step: normalize the description to ensure proper grammar
    description = normalize_description(description)

    return description
    