'''
Program to calculate total mAP on the selected dataset
'''
from CerebraGlossYOLO.model import CerebraGlossYOLO
from CerebraGlossYOLO.dataset import FPN_dataset
from CerebraGlossYOLO.utils import decode_predictions_fpn, batch_nms, decode_target_fpn
from torch.utils.data import DataLoader
import torch
from torchmetrics.detection import MeanAveragePrecision

def check_accuracy(loader: DataLoader, model: CerebraGlossYOLO, anchors_per_level: list[list[float]], nms_iou_threshold: float, conf_threshold=0.01) -> float:
    model.eval() # Ensure the model is in evaluation mode
    metric.reset() # Reset metric before each evaluation

    for x, y_list, attention_mask, pos_indices in loader:
        x = x.to(DEVICE)
        y_list = [y.to(DEVICE) for y in y_list]
        attention_mask = attention_mask.to(DEVICE)
        pos_indices = pos_indices.to(DEVICE)

        with torch.no_grad():
            predictions = model(x, attention_mask, pos_indices)
            batch_indices, channel_indices, b_x, b_w, scores, class_preds = \
                decode_predictions_fpn(predictions, anchors_per_level, conf_threshold=conf_threshold)
            
            preds = batch_nms(
                batch_indices, channel_indices, b_x, b_w, scores, class_preds, nms_iou_threshold, x.size(0)
            )
            targets = decode_target_fpn(y_list, anchors_per_level)
        
        metric.update(preds, targets)

    ret = metric.compute()
    return ret['map'].item()

VAL_FOLDERS = ['../CerebraGlossBench/npy'] 
MODEL_PATH = "./checkpoints/model.pkl"
CLASSES = ['sharp','spike','spsw','spindle','Kcomplex','eyem','eyer+','eyer-','hfnoise'] # 9 classes
# The following is consistent with training
NUM_CLASSES = len(CLASSES)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # NOTE change your GPU here
SEQ_LEN = 2000
ANCHORS_PER_LEVEL = [
    [90/SEQ_LEN,300/SEQ_LEN], # P3
    None, # P4
    None, # P5
    [1900/SEQ_LEN] # P6
]

ANCHORS_PER_LEVEL_NONONE = [a for a in ANCHORS_PER_LEVEL if a is not None]
ALL_S_LEVELS = [250, 125, 63, 32] # Number of S for each level
S_LEVELS = [s if ANCHORS_PER_LEVEL[i] is not None else None for i, s in enumerate(ALL_S_LEVELS)]
S_LEVELS_NONONE = [s for s in S_LEVELS if s is not None]
NUM_ANCHORS_PER_LEVEL = [len(anchors) if anchors is not None else None for anchors in ANCHORS_PER_LEVEL]

metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.5]).to(DEVICE) # mAP@0.5

val_dataset = FPN_dataset(
    S_LEVELS=S_LEVELS_NONONE, 
    anchors_per_level=ANCHORS_PER_LEVEL_NONONE, 
    folders=VAL_FOLDERS, 
    classes=CLASSES,
    use_augmentation=False,
    do_zscore = True
)
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=4, pin_memory=True)


print(f'Totally {len(val_dataset)} data, start checking')
model = CerebraGlossYOLO(num_classes=NUM_CLASSES,num_anchors_per_level=NUM_ANCHORS_PER_LEVEL).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
mean_avg_prec = check_accuracy(
    val_loader,
    model,
    ANCHORS_PER_LEVEL_NONONE,
    nms_iou_threshold=0.5,
)
print(f"mAP@0.5: {mean_avg_prec:.4f}")

