import torch
from typing import List


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Pure-Torch NMS matching torchvision.ops.nms for this CPU-safe demo.

    Boxes from different EEG channels occupy non-overlapping y intervals, so
    the ordinary 2-D IoU operation naturally performs channel-wise NMS.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = torch.argsort(scores, descending=True, stable=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        xx1 = torch.maximum(x1[current], x1[remaining])
        yy1 = torch.maximum(y1[current], y1[remaining])
        xx2 = torch.minimum(x2[current], x2[remaining])
        yy2 = torch.minimum(y2[current], y2[remaining])
        intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[current] + areas[remaining] - intersection
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        order = remaining[iou <= iou_threshold]
    return torch.stack(keep).to(dtype=torch.long)

def decode_predictions_fpn(predictions: List[torch.Tensor], anchors_per_level: List[List[float]], conf_threshold: float):
    """
    Decode FPN model outputs into a unified list of bounding boxes, keeping all boxes with confidence greater than conf_threshold.

    Args:
        predictions (List[Tensor[bs,C,S_level,B_level,3+num_class]]): List of outputs from each FPN prediction level
        anchors_per_level (List[List[float]]): Anchor widths for each level
        conf_threshold (float): Confidence threshold, predictions below this value will be ignored

    Returns:
        tuple: (batch_indices, channel_indices, b_x, b_w, scores, class_preds),
               (batch index, channel index, box center, box width, object confidence, object class).
               All are 1D tensors of the same length. batch_indices, channel_indices, and class_preds are integers; others are floats.
    """
    all_batch_indices, all_channel_indices = [], []
    all_b_x, all_b_w, all_scores, all_class_preds = [], [], [], []
    device = predictions[0].device

    for i, pred in enumerate(predictions):
        S = pred.size(2)
        anchors = torch.tensor(anchors_per_level[i], device=device, dtype=torch.float32)
        
        box_preds = pred[..., 0:2] # t_x, t_w (bs,C,S,B,2)
        box_conf = torch.sigmoid(pred[..., 2:3]) # confidence (bs,C,S,B,1)
        # class_probs = torch.softmax(pred[..., 3:], dim=-1) # (bs,C,S,B,2)
        class_probs = torch.sigmoid(pred[..., 3:]) # Match training loss function

        # Final confidence = box confidence * max class confidence
        max_class_probs, best_class_idx = torch.max(class_probs, dim=-1, keepdim=True) # (bs,C,S,B,1), force single label
        final_conf = box_conf * max_class_probs # (bs,C,S,B,1)

        # Filter predictions with low confidence
        mask = (final_conf > conf_threshold).squeeze(-1) # (bs,C,S,B)
        if mask.sum() == 0: # If no predictions pass the threshold
            continue

        # Use mask to select all relevant tensors
        t_x, t_w = box_preds[..., 0][mask], box_preds[..., 1][mask]

        # Get grid, anchor, batch, and channel indices for selected boxes
        batch_indices, channel_indices, grid_indices, anchor_indices = mask.nonzero(as_tuple=True)

        # Decode coordinates
        b_x = (t_x + grid_indices.float()) / S
        b_w = anchors[anchor_indices] * torch.exp(t_w)

        # Select confidence and class
        scores = final_conf[mask].squeeze(-1)
        class_preds = best_class_idx[mask].squeeze(-1)

        all_batch_indices.append(batch_indices)
        all_channel_indices.append(channel_indices)
        all_b_x.append(b_x)
        all_b_w.append(b_w)
        all_scores.append(scores)
        all_class_preds.append(class_preds)

    if not all_batch_indices: # If no boxes are detected in all levels
        return tuple(torch.tensor([], device=predictions[0].device) for _ in range(6))

    # Concatenate results from all levels
    return (
        torch.cat(all_batch_indices), torch.cat(all_channel_indices),
        torch.cat(all_b_x), torch.cat(all_b_w),
        torch.cat(all_scores), torch.cat(all_class_preds)
    )


def decode_target_fpn(targets: List[torch.Tensor], anchors_per_level: List[List[float]]):
    """
    Decode normalized ground truth bounding boxes from FPN target tensors.

    Args:
        targets (List[Tensor[bs,C,S_level,B_level,3+num_classes]]): Ground truth output from dataset
        anchors_per_level (List[List[float]]): Anchor widths for each FPN level

    Returns:
        List[dict[str,torch.Tensor]]: [{"boxes":Tensor[N,4], "labels":Tensor[N]}, ...], length is bs;
        For API compatibility, no scores are included (all scores are actually 1);
        For API compatibility, boxes use xyxy encoding, y1 is channel index, y2 = y1 + 1
    """
    all_boxes_by_batch = {} # {b_idx: [boxes_list]}
    all_labels_by_batch = {} # {b_idx: [labels_list]}
    device = targets[0].device
    bs = targets[0].size(0)

    for i, target in enumerate(targets):
        S = target.size(2)
        anchors = torch.tensor(anchors_per_level[i], device=device, dtype=torch.float32)
        
        # 1. Find all anchor positions containing objects (confidence == 1)
        obj_mask = (target[..., 2] == 1)
        if obj_mask.sum() == 0:
            continue

        # 2. Extract all relevant anchor data
        batch_indices, channel_indices, grid_indices, anchor_indices = obj_mask.nonzero(as_tuple=True)
        target_at_indices = target[batch_indices, channel_indices, grid_indices, anchor_indices]

        tx, tw = target_at_indices[:, 0], target_at_indices[:, 1] # (N,)
        class_probs = target_at_indices[:, 3:]

        # 3. Decode
        # Decode xyxy coordinates (normalized)
        b_x = (tx + grid_indices.float()) / S
        b_w = anchors[anchor_indices] * torch.exp(tw)
        xmin, xmax = b_x - b_w / 2, b_x + b_w / 2
        ymin, ymax = channel_indices.float(), channel_indices.float() + 1.0
        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        # Decode class
        labels = torch.argmax(class_probs, dim=1)

        # Group by batch
        for b_idx in range(bs):
            mask_b = (batch_indices == b_idx)
            if mask_b.sum() > 0:
                if b_idx not in all_boxes_by_batch:
                    all_boxes_by_batch[b_idx] = []
                    all_labels_by_batch[b_idx] = []
                all_boxes_by_batch[b_idx].append(boxes[mask_b])
                all_labels_by_batch[b_idx].append(labels[mask_b])
    
    # Organize into final output format
    output = []
    for b_idx in range(bs):
        if b_idx in all_boxes_by_batch:
            final_boxes = torch.cat(all_boxes_by_batch[b_idx])
            final_labels = torch.cat(all_labels_by_batch[b_idx])
            output.append({"boxes": final_boxes, "labels": final_labels})
        else:
            output.append({"boxes": torch.tensor([], device=device), "labels": torch.tensor([], device=device, dtype=torch.int64)})

    return output


def batch_nms(batch_indices:torch.Tensor,
              channel_indices:torch.Tensor,
              b_x:torch.Tensor,
              b_w:torch.Tensor,
              scores:torch.Tensor,
              class_preds:torch.Tensor,
              nms_iou_threshold:float,
              batch_size: int
            )->List[dict[str,torch.Tensor]]:
    """
    Perform NMS on the decoded results of decode_predictions, requires batch input.

    Args:
        nms_iou_threshold (float): IoU threshold, predictions with IoU below this are considered new boxes
        batch_size (int): Number of samples
        Other parameters (torch.Tensor): 1D tensors recording info for each bounding box in the batch

    Returns:
        List[dict[str,torch.Tensor]]: [{"boxes":Tensor[N,4], "scores":Tensor[N], "labels":Tensor[N]}, ...]
        List length is bs, can be indexed by batch_idx
        For API compatibility, boxes use xyxy encoding, y1 is channel index, y2 = y1 + 1
    """
    device = scores.device
    # Since decode_predictions_fpn may consider the last sample to have no bounding box, batch_indices.max() may not be batch_size
    # Therefore, batch_size must be passed in and should not be inferred from batch_indices
    
    # xyxy
    xmin = b_x - b_w / 2
    xmax = b_x + b_w / 2
    ymin = channel_indices.float()
    ymax = ymin + 1.0
    boxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

    output = []
    for b_idx in range(batch_size): 
        sample_mask = (batch_indices == b_idx)
        
        # Even if the sample has no predicted boxes, add an empty placeholder
        if not sample_mask.any():
            output.append({
                "boxes": torch.tensor([], device=device, dtype=torch.float32), 
                "scores": torch.tensor([], device=device, dtype=torch.float32), 
                "labels": torch.tensor([], device=device, dtype=torch.int64)
            })
            continue

        _boxes = boxes_xyxy[sample_mask]
        _scores = scores[sample_mask]
        _labels = class_preds[sample_mask]
        
        # keep_indices = torchvision.ops.batched_nms(_boxes, _scores, _labels, nms_iou_threshold)
        keep_indices = nms(_boxes, _scores, nms_iou_threshold) # Do not want multiple highly overlapping boxes with different labels in the same channel

        output.append({
            "boxes": _boxes[keep_indices],
            "scores": _scores[keep_indices],
            "labels": _labels[keep_indices].long(),
        })

    return output
