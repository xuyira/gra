'''
Find npy files from the specified folder to plot, create a new img directory under this folder to store the images. Single process.
'''
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


############
# Settings #
############
SFREQ = 200
display_channel_width = 200
NPY_DIR = Path('../npy')
chname = ["FP1-AV","FP2-AV","F3-AV","F4-AV","C3-AV","C4-AV","P3-AV","P4-AV","O1-AV","O2-AV","F7-AV","F8-AV","T3-AV","T4-AV","T5-AV","T6-AV","FZ-AV","CZ-AV","PZ-AV"]

image_output_folder = NPY_DIR / 'img'
image_output_folder.mkdir(parents=True, exist_ok=True)


########
# plot #
########
def plot_eeg_data(ax:Axes, data:np.ndarray, channel_names:list[str]):
    '''
    plot processed EEG npy files
    '''
    # EEG
    offset = np.arange(data.shape[0], 0, -1) * display_channel_width
    for i in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1])/SFREQ, -data[i]+offset[i], label=channel_names[i], color='black', linewidth=1)
        
    # 标尺的上下端点
    bar_y_start = offset[0] - display_channel_width / 2
    bar_y_end = offset[0] + display_channel_width / 2
    ax.plot([0,0], [bar_y_start, bar_y_end], color='red', linewidth=1, zorder=10)
    text_x_pos = -0.1
    ax.text(text_x_pos, bar_y_start, f"{display_channel_width} µV", 
            color='red', ha='right', va='center', fontsize=6)
    
    # legend
    ax.set_xticks(np.arange(0, data.shape[1]/SFREQ + 1, 1))
    ax.set_xlim(0,10)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_yticks(offset)
    ax.set_yticklabels([channel_name for channel_name in channel_names], fontsize=8)
    ax.set_ylim(display_channel_width//2, data.shape[0] * display_channel_width + display_channel_width//2)
    ax.set_xlabel("Time (s)", fontsize=8)


########
# CORE #
########
# iterate npy files
npy_files = sorted(list(NPY_DIR.glob('*.npy')))
for i, npy_file in enumerate(tqdm(npy_files)):
    data = np.load(npy_file)
    file_name = npy_file.name
    base_name = npy_file.stem

    assert len(chname) == data.shape[0], \
        f"Channel number mismatch for {file_name}: {len(chname)} vs {data.shape[0]}"

    # --- plot---
    num_channels = data.shape[0]
    fig_height = num_channels * 0.4
    fig, ax = plt.subplots(figsize=(6, fig_height))
    
    plot_eeg_data(ax, data, chname)
        

    image_save_path = image_output_folder / f'{base_name}.jpg'
    fig.savefig(image_save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

print("Processing complete.")