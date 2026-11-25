import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import torch
from CerebraGlossYOLO.model import CerebraGlossYOLO
from CerebraGlossYOLO.utils import decode_predictions_fpn, batch_nms
from background.background_calculate_FFT import extract_dominant_frequency_and_max_amplitude
from artifact.detect_artifact import EEGArtifact
from textGenerator.textGenerator_English import text_generator

#############################
# Main Processing Functions #
#############################
def waveform_detection(data:np.ndarray):
    """
    Waveform detection
    Args:j
        data(np.ndarray[len(channel_names),2000]): EEG data
    Return:
        bboxes(list[dict]): List of waveforms, format:
        [{"channel_idx","wave_type","confidence","time_range"},{"channel_idx","wave_type","confidence","time_range"}...]
    """
    global model,DEVICE,ANCHORS_PER_LEVEL_NONONE,CONF_THRESHOLD,NMS_IOU_THRESHOLD,SEQ_LEN,SAMPLING_RATE,CLASSES,STD_19_CHANNELS_CLEAN
    channel_names_clean = [ch.split('-')[0] for ch in channel_names]
    attention_mask = torch.tensor([ch not in channel_names_clean for ch in STD_19_CHANNELS_CLEAN], dtype=torch.bool)
    # pos_indices[i] represents the position of the i-th channel in eeg_tensor according to the 10-20 standard arrangement.
    pos_indices = torch.arange(19, dtype=torch.long)
    padded_data = np.zeros((19, SEQ_LEN), dtype=np.float32) # (19,2000)
    ch_idx_to_std_idx = [STD_19_CHANNELS_CLEAN.index(ch_name) for ch_name in channel_names_clean] # (C,2000) Channel names of current sample, corresponding to positions in standard arrangement
    # Fill (C,2000) into the standard 10-20 arrangement, (19,2000)
    for i in range(len(channel_names_clean)):
        padded_data[ch_idx_to_std_idx[i]] = data[i]

    data = np.expand_dims(padded_data, axis=0)  
    data = torch.from_numpy(data)
    # Normalization
    mean = data.mean(dim=2, keepdim=True)
    std = data.std(dim=2, keepdim=True)
    data = (data - mean) / (std + 1e-6)

    attention_mask = attention_mask.unsqueeze(0)  
    pos_indices = pos_indices.unsqueeze(0)  
    data = data.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    pos_indices = pos_indices.to(DEVICE)

    with torch.no_grad():
        predictions = model(data, attention_mask, pos_indices)
        batch_indices, channel_indices, b_x, b_w, scores, class_preds = \
            decode_predictions_fpn(predictions, ANCHORS_PER_LEVEL_NONONE, conf_threshold=CONF_THRESHOLD)
        preds = batch_nms(
            batch_indices, channel_indices, b_x, b_w, scores, class_preds, NMS_IOU_THRESHOLD, data.size(0)
        )
    # Modify bboxes
    cur_preds = preds[0]
    boxes = cur_preds["boxes"].to('cpu')
    confidence = cur_preds["scores"].to('cpu')
    labels = cur_preds["labels"].to('cpu')
    bboxes = []
    for box_idx in range(confidence.size(0)):
        x1, ch_idx, x2, _ = boxes[box_idx].tolist()
        ch = STD_19_CHANNELS_CLEAN[int(ch_idx)]
        ch_idx = [i for i, name in enumerate(channel_names_clean) if name == ch]
        start_sec = x1 * (SEQ_LEN / SAMPLING_RATE)
        end_sec = x2 * (SEQ_LEN / SAMPLING_RATE)
        bboxes.append({
            "channel_idx": ch_idx[0],
            "wave_type": CLASSES[int(labels[box_idx].item())],
            "confidence": confidence[box_idx].item(),
            "time_range": [max(0, start_sec), min(SEQ_LEN/SAMPLING_RATE, end_sec)]
        })
    return bboxes

def background_detection(data_path):
    """
    Background detection
    Args:
        data_path(str): Filename of EEG npy data
    """
    dominant_freq_matrix, max_amp_matrix = extract_dominant_frequency_and_max_amplitude(data_path,SAMPLING_RATE)
    return dominant_freq_matrix, max_amp_matrix

def artifact_detection(data):
    """
    Artifact detection
    """
    global channel_names
    clean_ch_order = []
    for ch in channel_names:
        base_name = ch.split('-')[0]
        if base_name.upper().startswith('FP'):
            clean_name = 'Fp' + base_name[2:]
        else:
            clean_name = base_name
        clean_ch_order.append(clean_name)

    artifact_model = EEGArtifact(data=data, freq=SAMPLING_RATE, ch_order=clean_ch_order, verbose=False)
    artifact_dict = artifact_model.get_annotations_dict()
    
    return artifact_dict

def text_generate(data,data_path):
    """
    Text generator
    Args:
        data(np.ndarray[len(channel_names),2000]): EEG data
    Return:
        text(str): Analysis text for EEG
    """
    global channel_names
    bbox = waveform_detection(data)
    dominant_freq_matrix, max_amp_matrix = background_detection(data_path)
    artifact_dict = artifact_detection(data)
    artifact_bboxes = artifact_dict['bboxes']
    
    text = text_generator(bbox, artifact_bboxes, dominant_freq_matrix, max_amp_matrix, channel_names)
    return text

######################
# Plotting Functions #
######################
def plot_eeg_data(data:np.array,scale,time_scale):
    '''
    Plot EEG npy file, initialize fig, ax
    Args:
        data(np.ndarray[len(channel_names),2000]): EEG data
        scale: amplitude scaling factor
        time_scale: time scaling factor
    '''
    global fig,ax,display_channel_width,offset,figure_legends,channel_names,wavetype_patches, background_patches, artifact_patches
    if data.shape[0] <= 10:
        fig, ax = plt.subplots(figsize=(15*time_scale,5))
    else:
        fig, ax = plt.subplots(figsize=(15*time_scale,10))
    offset = np.arange(data.shape[0], 0, -1) * display_channel_width
    # Plot EEG
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1])/SAMPLING_RATE, -data[i]*scale+offset[i], label=channel_names[i], color='black', linewidth=1)
    ax.set_xticks(np.arange(0, data.shape[1]/SAMPLING_RATE + 1, data.shape[1]/SAMPLING_RATE//10)) 
    ax.set_xticklabels(np.arange(0, 10 + 1, 1))
    ax.set_yticks(offset)
    ax.set_yticklabels([channel_name for channel_name in channel_names])
    ax.set_ylim(display_channel_width//2, data.shape[0] * display_channel_width + display_channel_width//2)
    ax.set_xlabel("Time (s)")
    # Delete existing legends
    figure_legends = []
    wavetype_patches= []
    background_patches= []
    artifact_patches=[]
    return

def plot_boxes(bboxes):
    '''
    Add bounding boxes to fig
    '''
    global fig,offset,box_colors,display_channel_width,channel_names
    ax = fig.axes[0]
    # Remove existing bounding boxes
    for text in ax.texts:
        text.remove()
    for collection in ax.collections:
        collection.remove()
    # Draw the bounding box using coloring.
    for bbox in bboxes:
        ax.fill_between(x=bbox['time_range'],
                        y1=offset[bbox['channel_idx']]-display_channel_width//2.5, 
                        y2=offset[bbox['channel_idx']]+display_channel_width//2.5,
                        alpha=0.3, edgecolor='none',
                        color=box_colors[bbox['wave_type']]
                    ) 
        ax.text(
                x = bbox['time_range'][0],
                y = offset[bbox['channel_idx']]+display_channel_width//2.5,
                s = round(bbox['confidence'],2)
        )
        if bbox['wave_type'] not in wavetype_patches:
            wavetype_patches.append(bbox['wave_type'])
    # Update legend
    add_or_delete_patches(box_colors, wavetype_patches,True)
    return

def plot_background(data,scale,dominant_freq_matrix, max_amp_matrix):
    '''
    Add background to fig
    '''
    global fig,ax,display_channel_width,background_patches,offset,background_colors
    ax = fig.axes[0]
    for line in ax.lines[:]:
        line.remove()
    for i in range(dominant_freq_matrix.shape[0]):
        for j in range(dominant_freq_matrix.shape[1]):
            if 8 <= dominant_freq_matrix[i][j] < 14: 
                background = 'α (8-14Hz)'
            elif 14 <= dominant_freq_matrix[i][j] < 30: 
                background = 'β (14-30Hz)'
            elif 4 <= dominant_freq_matrix[i][j] < 8: 
                background = 'θ (4-8Hz)'
            elif 0.3 < dominant_freq_matrix[i][j] < 4:
                background = 'δ (0.3-4Hz)'
            elif 30 <= dominant_freq_matrix[i][j] < 70: 
                background = 'γ (30-70Hz)'
            else: 
                background = 'others'
            if background not in background_patches:
                background_patches.append(background)
            ax.plot((np.arange(data.shape[1]/10)+ j*200)/SAMPLING_RATE, -data[i][200*j:200*(j+1)]*scale+offset[i], color=background_colors[background], linewidth=1)
    add_or_delete_patches(background_colors, background_patches,True)
    return

def plot_artifact(artifact_dict):
    '''
    Add artifacts to fig
    '''
    global fig,offset,display_channel_width,artifact_color,artifact_box_width
    ax = fig.axes[0]
    for artifact in artifact_dict['bboxes'][:]:
        rect = patches.Rectangle(
            (artifact['time_range'][0], offset[artifact['channel_idx']]-artifact_box_width[artifact['wave_type']]/2),  # Bottom-left coordinate (x, y)
            (artifact['time_range'][1] - artifact['time_range'][0]),  # Width
            artifact_box_width[artifact['wave_type']], # Height
            linewidth=1, # Border width
            edgecolor=artifact_color[artifact['wave_type']], # Border color
            facecolor='none', # Key: no fill
            alpha=0.7    # Transparency
        )
        ax.add_patch(rect)
        if artifact['wave_type'] not in artifact_patches:
            artifact_patches.append(artifact['wave_type'])
    add_or_delete_patches(artifact_color,artifact_patches,True)
    return

def add_or_delete_patches(colors:dict,labels:list,add_flag=True):
    '''
    Add or remove legends in fig
    Args:
        colors(dict): mapping of legend to color
        labels(list): waveform/background/artifact names to add to legend
        add_flag(bool): if True, add legend; else remove legend
    '''
    global fig,figure_legends
    ax = fig.axes[0]
    existing_labels = {p.get_label() for p in figure_legends}
    if add_flag:
    # Add legend
        figure_legends.extend([mpatches.Patch(color=colors[wave_type], alpha=0.3, label=wave_type) for wave_type in labels
                if wave_type not in existing_labels])
    else:
    # Remove legend
        figure_legends[:] = [p for p in figure_legends if p.get_label() not in labels]
    # Remove original legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # Add new legend
    ax.legend(handles=figure_legends, 
                loc='upper left',
                bbox_to_anchor=(1, 1), # Align the point to the outside of the top right corner of the coordinate axis.
                frameon=True
            )
    fig.tight_layout()
    return

###################
# Basic Functions #
###################
def get_data(file,ch_order_str=None):
    '''
    Get ndarray data from file, initialize channel_names in this function
    Args:
        file:
        ch_order_str(str): Channels in EEG data, separated by ',', should be in standard channels. TU and Dreams datasets do not require this parameter
        Example: 'FP1-AV,CZ-AV,O1-AV'
    Returns:
        data(ndarray): EEG data
        msg(str): Information to write to status
    '''
    global channel_names
    # Read npy file
    try:
        data = np.load(file.name)
    except Exception:
        return None,''
    else:
        if (ch_order_str is None )or(ch_order_str == ""):
            if data.shape[0] == 19:
                # TU dataset
                channel_names = STD_19_CHANNELS
            elif data.shape[0] == 3:
                # Dreams dataset
                channel_names = ["FP1-A2","CZ-A1","O1-A2"]
            else:
                return None, "Non-predefined data, please input its channel order"
        else:
            ch_clean = [ch.strip() for ch in ch_order_str.split(',') if ch.strip()]
            channel_names = [ch for ch in ch_clean if ch.split('-')[0] in STD_19_CHANNELS_CLEAN]
           
    return data, "Data loaded successfully!"

def delete_sth(name, data=None,scale=1.0):
    '''
    Used to delete specified content from the drawn fig
    '''
    global fig, wavetype_patches, background_patches, artifact_patches
    if name == 'waveform':
        ax = fig.axes[0]
        for coll in ax.collections[:]:
            coll.remove()
        for text in ax.texts:
            text.remove()
        add_or_delete_patches(box_colors, wavetype_patches,False)
        wavetype_patches = []
    elif name == 'background':
        ax = fig.axes[0]
        for line in ax.lines[:]:
            line.remove()
        for i in range(data.shape[0]):
            ax.plot(np.arange(data.shape[1])/SAMPLING_RATE, -data[i]*scale+offset[i], color='black', linewidth=1)
        add_or_delete_patches(background_colors,background_patches,False)
        background_patches = []
    elif name =='artifact':
        ax = fig.axes[0]
        for patch in ax.patches[:]:
            if isinstance(patch, patches.Rectangle):  # Delete rectangles only
                patch.remove()
        add_or_delete_patches(artifact_color,artifact_patches,False)
        artifact_patches = []
    else:
        return

#################################################
#   Final processing function for each button   #
#################################################
def scale_slider_function(file, scale, time_scale, ch_order=None):
    '''
    Handler for time_scale_slider and voltage_scale_slider
    '''
    data,msg = get_data(file,ch_order)
    if data is None:
        return None
    if event_done['background_button'] is True:
        if data.shape[0] <= 10:
            fig.set_size_inches(15*time_scale, 5)
        else:
            fig.set_size_inches(15*time_scale, 10)
        ax = fig.axes[0]
        dominant_freq_matrix, max_amp_matrix = background_detection(file.name)
        plot_background(data,scale,dominant_freq_matrix, max_amp_matrix)
    else:
        if data.shape[0] <= 10:
            fig.set_size_inches(15*time_scale, 5)
        else:
            fig.set_size_inches(15*time_scale, 10)
        ax = fig.axes[0]
        for line in ax.lines[:]:
            line.remove()
        for i in range(data.shape[0]):
            ax.plot(np.arange(data.shape[1])/SAMPLING_RATE, -data[i]*scale+offset[i], color='black', linewidth=1)
        fig.tight_layout()
    return fig

def waveform_button(file,ch_order=None):
    '''
    Handler for waveform_btn
    '''
    global fig,event_done
    if event_done['waveform_button'] is False:
        data, msg = get_data(file,ch_order)
        if data is None:
            return None, msg, gr.update(value="Waveform Detection")
        bbox = waveform_detection(data)
        plot_boxes(bbox)
        event_done['waveform_button'] = True
        return fig, "Waveform drawing completed", gr.update(value="Undo Waveform")
    else:
        delete_sth('waveform')
        event_done['waveform_button'] = False
        return fig, "Waveform undone", gr.update(value="Waveform Detection")

def background_button(file,scale=1.0,ch_order=None):
    '''
    Handler for bg_btn
    '''
    global fig,event_done
    if event_done['background_button'] is False:
        data,msg = get_data(file,ch_order)
        if data is None:
            return None, msg, gr.update(value="Background Detection")
        dominant_freq_matrix, max_amp_matrix = background_detection(file.name)
        plot_background(data,scale,dominant_freq_matrix, max_amp_matrix)
        event_done['background_button'] = True
        return fig, "Background drawing completed", gr.update(value="Undo Background")
    else:
        data,msg = get_data(file,ch_order)
        delete_sth('background',data,scale)
        event_done['background_button'] = False
        return fig, "Background undone", gr.update(value="Background Detection")

def artifact_button(file,ch_order=None):
    '''
    Handler for artifact_btn
    '''
    global fig, event_done
    if event_done['artifact_button'] is False:
        data,msg = get_data(file,ch_order)
        if data is None:
            return None, msg, gr.update(value="Artifact Detection")
        artifact_dict = artifact_detection(data)
        plot_artifact(artifact_dict)
        event_done['artifact_button'] = True
        return fig, "Artifact drawing completed", gr.update(value="Undo Artifact")
    else:
        delete_sth('artifact')
        event_done['artifact_button'] = False
        return fig, "Artifact undone", gr.update(value="Artifact Detection")

def text_button(file,ch_order):
    '''
    Handler for report_btn
    '''
    data,msg = get_data(file,ch_order)
    if event_done['text_button'] is False:
        data,msg = get_data(file,ch_order)
        if data is None:
            return '', gr.update(value="Report Output")
        text = text_generate(data, file.name)
        event_done['artifact_button'] = True
        return text, gr.update(value="Report Completed")
    else:
        text = text_generate(data, file.name)
        return text, gr.update(value="Report Output")

def process_file(file,scale=1.0,time_scale=1.0,ch_order=None):
    '''
    Handler for file_input
    '''
    global fig
    data, msg = get_data(file,ch_order)
    if data is None:
        button_flag_init()
        return None, '', "no data", gr.update(value="Waveform Detection"), gr.update(value="Background Detection"), gr.update(value="Artifact Detection"), gr.update(value="Report Output")
    # Plot original EEG
    plot_eeg_data(data, scale,time_scale)
    button_flag_init()
    return fig,'', msg, gr.update(value="Waveform Detection"), gr.update(value="Background Detection"), gr.update(value="Artifact Detection"), gr.update(value="Report Output")


############
# Settings #
############
def model_init():
    '''
    YOLO model initialization
    '''
    global model,DEVICE,ANCHORS_PER_LEVEL_NONONE,CONF_THRESHOLD,NMS_IOU_THRESHOLD,SEQ_LEN,SAMPLING_RATE,CLASSES,STD_19_CHANNELS,STD_19_CHANNELS_CLEAN
    MODEL_DIR = "."
    # Configuration parameters, must be consistent with training
    SEQ_LEN = 2000
    SAMPLING_RATE = 200
    ANCHORS_PER_LEVEL = [
        [90/SEQ_LEN,300/SEQ_LEN], # P3
        None, # P4
        None, # P5
        [1900/SEQ_LEN] # P6
    ]

    CLASSES = ['sharp','spike','spsw','spindle','Kcomplex','eyem','eyer+','eyer-','hfnoise'] # 9cls
    NUM_CLASSES = len(CLASSES)
    STD_19_CHANNELS = ['FP1-AV', 'FP2-AV', 'F3-AV', 'F4-AV', 'C3-AV', 'C4-AV',  
               'P3-AV',  'P4-AV', 'O1-AV', 'O2-AV', 'F7-AV', 'F8-AV', 
               'T3-AV',  'T4-AV', 'T5-AV', 'T6-AV', 'FZ-AV', 'CZ-AV', 'PZ-AV']
    STD_19_CHANNELS_CLEAN = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',  
                'P3',  'P4', 'O1', 'O2', 'F7', 'F8', 
                'T3',  'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
    # The following parameters do not need to be modified
    ANCHORS_PER_LEVEL_NONONE = [a for a in ANCHORS_PER_LEVEL if a is not None] # [[0.045, 0.15], [0.95]]
    ALL_S_LEVELS = [250, 125, 63, 32] # Number of S per level
    NUM_ANCHORS_PER_LEVEL = [len(anchors) if anchors is not None else None for anchors in ANCHORS_PER_LEVEL] # [2, None, None, 1]
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # NOTE change your GPU here
    # Inference hyperparameters
    MODEL_PATH = f"{MODEL_DIR}/checkpoints/model.pkl"
    CONF_THRESHOLD = 0.5  # Confidence threshold, prediction boxes below this value will be ignored.
    NMS_IOU_THRESHOLD = 0.5 # IoU threshold of NMS
    model = CerebraGlossYOLO(num_classes=NUM_CLASSES,num_anchors_per_level=NUM_ANCHORS_PER_LEVEL).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return 

def fig_init():
    '''
    Initialization of plotting parameters
    '''
    global box_colors,background_colors, display_channel_width,fig,artifact_color,wavetype_patches,background_patches,artifact_patches,figure_legends,artifact_box_width
    display_channel_width = 150
    artifact_box_width = {
        "severe_artifact":display_channel_width/1.05, "nan_inf": display_channel_width/1, "global_bad": display_channel_width/1,
        "flat": display_channel_width/1.2, "muscle":display_channel_width/1.3, 
        "eog_v": display_channel_width/1.1, "eog_left": display_channel_width/1.1, "eog_right":display_channel_width/1.1, "respiration": display_channel_width/1.4,
        "drowsiness":display_channel_width/1.45
    }
    box_colors = {
                    'sharp':'#8B0000','spike':"#21C021",'spsw':'#00008B','alpha':'#FFD700',
                    'delta':'#FF00FF','spindle':'#FF8C00','Kcomplex':'#4B0082','eyem':"#3551DE",
                    'eyer+':"#800080",'eyer-':"#D2B48C",'hfnoise':'#008B8B'
                }
    background_colors = {
        'α (8-14Hz)': '#d62728',
        'β (14-30Hz)': '#ff7f0e',
        'θ (4-8Hz)': '#2ca02c',
        'δ (0.3-4Hz)': '#1f77b4',
        'γ (30-70Hz)': '#9467bd',
        'others':'black'
    }
    artifact_color = {
        "severe_artifact": '#DC143C', "nan_inf": "#808000", "global_bad": '#4169E1',
        "flat": "#D2691E", "muscle": '#4682B4', 
        "eog_v": "#FF1493", "eog_left": '#006400', "eog_right": "#00BFFF", "respiration": '#8B008B',
        "drowsiness": '#FF7F50'
    }

    wavetype_patches = []
    background_patches = []
    artifact_patches = []
    figure_legends = []

def button_flag_init():
    '''
    Initialize buttons to unpressed state
    '''
    global event_done
    event_done={}
    event_done['artifact_button'] = False
    event_done['waveform_button'] = False
    event_done['background_button'] = False
    event_done['text_button'] = False


# Initialization
model_init()
fig_init()
button_flag_init()
# UI
with gr.Blocks(
    # --- Use this final version of CSS ---
    title="EEG Signal Analysis",
    css="""
    /* Add bottom padding to content area to prevent toolbar overlap */
    .content-column {
        padding-bottom: 300px !important;  /* Adjust according to toolbar height */
    }
    
    /* Ensure toolbar is fixed at the bottom */
    #fixed-toolbar {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        background: #f8f8f8;
        z-index: 999;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.08);
        padding: 10px 0;
    }
    
    .second-row {
        margin-top: 10px;
    }

    /* --- File upload box style remains unchanged --- */
    #my-file-input {
        height: 90px !important;
        min-height: 32px !important;
    }
    
    #my-file-input > button {
        display: flex !important;
        width: 100%;
        height: 100%;
        align-items: center !important;
        justify-content: center !important;
    }
    
    #my-file-input > button > svg, 
    #my-file-input > button > div {
        display: none !important;
    }
    
    #my-file-input > button::after {
        content: 'Drag or click to upload .npy file';
        color: gray;
        font-size: 1.2em;
        text-align: center;
    }
    """
    # ------------------------------------
    ) as app:
    # Output area
    with gr.Column(elem_classes="content-column"):
        plot_output = gr.Plot(label="Image Output")
        text_output = gr.Textbox(label="Report Output", lines=10)
        status = gr.Textbox(label="Output Log", interactive=False)
        

    # Fixed bottom toolbar - changed to Column container with two Rows
    with gr.Column(elem_id="fixed-toolbar"):
        # First row: only file input
        with gr.Row():
            file_input = gr.File(label="Upload .npy file", file_types=[".npy"], elem_id="my-file-input")
            ch_order = gr.Textbox(label="ch_order")
        
        # Second row: all other components
        with gr.Row(elem_classes="second-row"):
            waveform_btn = gr.Button("Waveform Detection")
            bg_btn = gr.Button("Background Detection")
            artifact_btn = gr.Button("Artifact Detection")
            report_btn = gr.Button("Report Output")
            voltage_scale_slider = gr.Slider(label="Amplitude Scale", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
            time_scale_slider = gr.Slider(label="Time Scale", minimum=0.5, maximum=1.2, value=0.7, step=0.05)

    # Bind events
    file_input.change(
        fn=process_file,
        inputs=[file_input, voltage_scale_slider,time_scale_slider,ch_order],
        outputs=[plot_output, text_output,status, waveform_btn, bg_btn, artifact_btn, report_btn]
    )
    voltage_scale_slider.change(
        fn=scale_slider_function,
        inputs=[file_input, voltage_scale_slider,time_scale_slider,ch_order],
        outputs=plot_output
    )
    time_scale_slider.change(
        fn=scale_slider_function,
        inputs=[file_input, voltage_scale_slider,time_scale_slider,ch_order],
        outputs=plot_output
    )
    waveform_btn.click(
        fn=waveform_button,
        inputs=[file_input,ch_order],
        outputs=[plot_output, status, waveform_btn]
    )
    bg_btn.click(
        fn=background_button,
        inputs=[file_input, voltage_scale_slider,ch_order],
        outputs=[plot_output, status, bg_btn]
    )
    
    artifact_btn.click(
        fn=artifact_button,
        inputs=[file_input,ch_order],
        outputs=[plot_output, status, artifact_btn]
    )
    
    report_btn.click(
        fn=text_button,
        inputs=[file_input,ch_order],
        outputs=[text_output, report_btn]
    )

# Launch app
if __name__ == "__main__":
    app.launch()