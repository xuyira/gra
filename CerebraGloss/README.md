# CerebraGloss

The official repository for our ICLR 2026 paper [CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation](https://openreview.net/forum?id=Xi1jkajWi9).

## Abstract

Interpreting clinical electroencephalography (EEG) is a laborious, subjective process, and existing computational models are limited to narrow classification tasks rather than holistic interpretation. A key bottleneck for applying powerful Large Vision-Language Models (LVLMs) to this domain is the scarcity of datasets pairing EEG visualizations with fine-grained, expert-level annotations. We address this by introducing CerebraGloss, an instruction-tuned LVLM for nuanced EEG interpretation. We first introduce a novel, automated data generation pipeline, featuring a bespoke YOLO-based waveform detector, to programmatically create a large-scale corpus of EEG-text instruction data. Using this data, we develop CerebraGloss, the first model of its kind capable of unified, generative analysis—performing tasks from detailed waveform description to multi-turn, context-aware dialogue. To evaluate this new capability, we construct and release CerebraGloss-Bench, a comprehensive benchmark for open-ended EEG interpretation. CerebraGloss demonstrates strong performance, surpassing leading LVLMs, including proprietary models like GPT-5, on this benchmark and achieving a new state-of-the-art on the TUSZ seizure detection task.

## TL;DR

**Build EEG-text Pairs**

- Data engine
    - CerebraGloss-YOLO: spike, sharp, spike/sharp-and-wave, sleep spindle, K-complex, blink, lateral eye movement, high frequency noise
    - Background: frequency and amplitude
    - Artifact: blink, lateral eye movement, high frequency noise, low correlation, flat, respiration, extreme values
- Rules and Gemini 2.5 Flash to build description, MCQ, conversation

**Post-training**

SFT Qwen2.5-VL to understand EEG (fine-grained events, sleep and epilepsy)

- Stage 1: SFT projector with captions
- Stage 2: SFT projector and LLM decoder with instruction-following dataset

**Downstream**

- CerebraGloss-Bench: 90 samples with description, MCQ, conversation and bounding boxes
- TUSZ (seizure) and HMC (sleep stage)

## How to Start

To ensure the reproducibility of our work, we are committed to releasing the core components of our project. This includes the code for our programmatic data engine, the model weights for CerebraGloss, and the complete CerebraGloss-Bench.

This folder provides three demos, covering our data engine, CerebraGloss and CerebraGloss-Bench. Please refer to the `REAMDE.md` files in corresponding folders to get more information.

Moreover, we provide `intro_en.mp4` to help you get a preliminary understanding of the demo.

## Citation

```
@inproceedings{
gu2026cerebragloss,
title={CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical {EEG} Interpretation},
author={Wei Gu and Tianming Luo and Qiran Zhang and Mohan Ye and Xiao Shen and Wenxin Chen and Yunhuan Li and Yichen Zhang and Jing Hong and Bao-liang Lu and Wei-Long Zheng},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Xi1jkajWi9}
}
```