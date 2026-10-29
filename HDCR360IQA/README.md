# HDCR360IQA

HDCR360IQA is a no-reference image quality assessment model for 360-degree (omnidirectional) images. It adopts the Swin Transformer as the backbone and integrates a **Distortion-Aware Feature Modulation (DAFM)** module and a **Wavelet Enhancement (WavE)** module to capture distortion-sensitive and high-frequency degradation cues on equirectangular panoramas.

The model supports three commonly used omnidirectional IQA datasets:

- **CVIQD**
- **OIQA**
- **IQA-ODI**

## Installation

```bash
conda create -n hdcr360 python=3.9 -y
conda activate hdcr360
```

Install PyTorch 2.4 with CUDA 12 (or higher):

```bash
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:

```bash
pip install opencv-python numpy scipy tqdm tensorboard
```

## Dataset Preparation

1. Download the CVIQD, OIQA, and/or IQA-ODI datasets.
2. Place the distorted images and the corresponding label files (`.txt` / `.csv`) following the structure under `data/`:

```
data/
├── cviqd/
│   └── cviqd_label_7.py
├── OIQA/
│   ├── oiqa_label_9.py
│   ├── oiqa_dis_label.txt
│   └── oiqa_ref_label.txt
└── IQA_ODI/
    ├── iqaodi_label_6.py
    ├── iqaodi_ref_dis_label.txt
    └── odi_iqa_info.csv
```

3. Download the Swin-Tiny pretrained weights (`swin_tiny_patch4_window7_224.pth`) from the official [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer).

## Usage

Open `train.py`, update the dataset paths, label paths, and the pretrained weight path in the `Config`, then select the target dataset via `dataset_name`:

```python
"dataset_name": "iqaodi",   # options: "cviqd", "oiqa", "iqaodi"
```

Start training:

```bash
python train.py
```

Checkpoints, logs, and TensorBoard records are saved under `./outputc/`:

```bash
tensorboard --logdir ./outputc/tensorboard
```

## Project Structure

```
├── train.py              # Training / validation entry point
├── load_train_t.py       # Training and evaluation loops
├── swin/                 # Swin Transformer backbone and the HDCR360IQA model
├── data/                 # Dataset loaders and label files
├── cviqd/                # CVIQD label generation utilities
└── utils_tools/          # ERP-to-viewport projection and preprocessing tools
```

## License

This project is released under the [MIT License](LICENSE).
