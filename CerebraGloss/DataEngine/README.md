# Data Engine

Our data engine includes **Key Waveform Event Detection**, **Background Rhythm Characterization**, **Artifact Identification** and **Rule-Based Caption Generation**. It converts raw EEG data to fine-grained caption.
This folder contains a demo to help you understand how it works.

## Environment

Our data engine is built under torch==2.7.0, torchvision==0.22.0 and python==3.12.9. Besides, you also need to install h5py, numpy, matplotlib, scipy, einops, gradio and torchmetrics.
The data engine should be version-agnostic to the various packages.

## Run

Under `DataEngine` folder, run:
- `python dataEngine_demo.py` to see how our data engine converts raw EEG data to fine-grained caption. `npy4test` folder contains npy examples for you to reproduce the visualization result of CerebraGloss-YOLO in the paper.
- `python loadtest.py` to reproduce the mAP results of CerebraGloss-YOLO in the paper.
