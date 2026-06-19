# MambaVAD
Official PyTorch implementation of the manuscript "MambaVAD: Learning Temporal Dynamics for Weakly Supervised Video Anomaly Detection via Bidirectional Mamba with Future-Score Alignment".

## Project Overview
MambaVAD is a weakly supervised video anomaly detection (WSVAD) framework based on the Mamba State Space Model. The core idea is to leverage a **bidirectional Mamba encoder** to capture both forward and backward temporal dependencies in video sequences, enabling more effective anomaly detection. The model takes pre-extracted CLIP features as input and applies multi-scale temporal fusion, normal prototype contrastive learning, and future score prediction for robust anomaly detection.

## Features
- **Bidirectional Mamba Encoder**: Forward and backward scan with concatenation and projection to capture bidirectional temporal context
- **Multi-scale Temporal Fusion**: Optional multi-scale feature extraction with configurable scale factors (e.g., 1, 8, 16)
- **Normal Prototype Contrastive Loss**: Encourages compact normal feature representation and separates anomaly features from normal prototypes
- **Future Score Prediction Bridge**: Auxiliary task that predicts future classification scores from hidden states
- **Mamba1/Mamba2 Support**: Configurable Mamba version via `--mamba-version`
- **Support for UCF-Crime and XD-Violence Datasets**

## Project Structure
```
.
├── list/                         # Ground truth generation and data list scripts
│   ├── make_gt_mAP_ucf.py
│   ├── make_gt_mAP_xd.py
│   ├── make_gt_ucf.py
│   ├── make_gt_xd.py
│   ├── make_list_ucf.py
│   └── make_list_xd.py
├── src_sgl/                      # Single bidirectional Mamba VAD implementation
│   ├── clip/                     # CLIP model components
│   │   ├── __init__.py
│   │   ├── clip.py
│   │   ├── model.py
│   │   └── simple_tokenizer.py
│   ├── utils/                    # Utility functions
│   │   ├── dataset.py            # UCFDataset and XDDataset loaders
│   │   ├── layers.py
│   │   ├── lr_warmup.py
│   │   ├── tools.py              # Feature processing, prompt text, batch labels
│   │   ├── ucf_detectionMAP.py
│   │   └── xd_detectionMAP.py
│   ├── mamba_model.py            # Mamba building blocks (MambaResBlock, BiMambaEncoder, etc.)
│   ├── mamba_vad_dual_model.py   # SingleMambaVAD model (full architecture)
│   ├── ucf_option.py             # UCF-Crime dataset options/hyperparameters
│   ├── ucf_train_sgl.py          # UCF-Crime training script
│   ├── ucf_test_sgl.py           # UCF-Crime testing script
│   ├── xd_option.py              # XD-Violence dataset options/hyperparameters
│   ├── xd_train_sgl.py           # XD-Violence training script
│   └── xd_test_sgl.py            # XD-Violence testing script
└── readme.md
```

## Model Architecture
```
CLIP Features → Position Embedding → BiMambaEncoder (or MambaTemporalEncoder)
                                    → Multi-scale Fusion (optional)
                                    → MLP Classifier → Anomaly Scores
```

**Training Objective:**
```
L = L_cls + α * L_proto + β * L_future
```
- `L_cls`: Binary cross-entropy classification loss with top-k pooling
- `L_proto`: Normal prototype contrastive loss
- `L_future`: Future score prediction MSE loss

## Installation

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA-capable GPU

### Install Dependencies
```bash
pip install torch torchvision
pip install mamba-ssm causal-conv1d --no-build-isolation
pip install numpy pandas scikit-learn
```

## Datasets
The project supports two benchmark datasets:

- **UCF-Crime**: 13 anomaly classes + normal, large-scale real-world anomaly detection dataset
- **XD-Violence**: 6 violence categories + normal, extreme domain violence detection dataset

### Setup
The extracted CLIP features for UCF-Crime and XD-Violence datasets can be obtained from the following links:

| Benchmark   | CLIP [Baidu]                                                                                          | CLIP [OneDrive]                                                                                                                              |
|-------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| UCF-Crime   | [Code: 7yzp](https://pan.baidu.com/s/1OKRIxoLcxt-7RYxWpylgLQ)                                       | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Ea86YOcp5z9KhRFDQm9a8zwBcGiGGg5BuBJtgmCVByazBQ?e=tqHLHt) |
| XD-Violence | [Code: v8tw](https://pan.baidu.com/s/1q8DiYHcPJtrBQiiJMI7aJw)                                        | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/Et5dWQZb2cBDs7zsrp90SrQBL_52vTRNYTdjQW6SMl0ZVA?e=foX4ph) |

After downloading, update the following files to point to your local dataset paths:
- `list/ucf_CLIP_rgb.csv` and `list/ucf_CLIP_rgbtest.csv` for UCF-Crime
- `list/xd_CLIP_rgb.csv` and `list/xd_CLIP_rgbtest.csv` for XD-Violence

## Usage

### Training

#### UCF-Crime
```bash
cd src_sgl
python ucf_train_sgl.py --bidirectional --max-epoch 10 --batch-size 64 --lr 2e-5
```

#### XD-Violence
```bash
cd src_sgl
python xd_train_sgl.py --bidirectional --max-epoch 10 --batch-size 96 --lr 1e-5
```

### Testing

#### UCF-Crime
```bash
cd src_sgl
python ucf_test_sgl.py --model-path model/model_ucf.pth --bidirectional
```

#### XD-Violence
```bash
cd src_sgl
python xd_test_sgl.py --model-path model/model_xd.pth --bidirectional
```

### Key Arguments

| Argument               | Default       | Description                                                        |
|------------------------|---------------|--------------------------------------------------------------------|
| `--bidirectional`      | `False`       | Enable bidirectional Mamba encoder (forward + backward scan)       |
| `--multi-scale`        | `False`       | Enable multi-scale temporal fusion                                 |
| `--multi-scale-factors`| `1,8,16`      | Scale factors for multi-scale fusion                               |
| `--mamba-version`      | `mamba1`      | Mamba version (`mamba1` or `mamba2`)                               |
| `--mamba-layers`       | `2` (UCF) / `1` (XD) | Number of Mamba layers                                     |
| `--d-state`            | `16`          | Mamba state dimension                                                |
| `--d-conv`             | `4`           | Mamba convolution kernel size                                      |
| `--expand`             | `2`           | Mamba expansion factor                                             |
| `--mlp-scale`          | `4`           | MLP hidden dimension scale factor                                  |
| `--visual-length`      | `256`         | Input sequence length (number of segments)                         |
| `--visual-width`       | `512`         | Feature dimension (CLIP feature dim)                               |
| `--max-epoch`          | `10`          | Maximum training epochs                                            |
| `--batch-size`         | `64` (UCF) / `96` (XD) | Training batch size                                       |
| `--lr`                 | `2e-5` (UCF) / `1e-5` (XD) | Learning rate                                         |
| `--mamba-lr-scale`     | `1.0`         | Learning rate scale for Mamba parameters (< 1 for lower Mamba LR)  |
| `--label-smooth`       | `0.0`         | Label smoothing factor for classification loss                     |
| `--weight-decay`       | `0.0`         | Weight decay for optimizer                                         |
| `--dropout`            | `0.0`         | Dropout rate in Mamba blocks                                       |
| `--model-path`         | `model/model_ucf.pth` / `model/model_xd.pth` | Path to saved model checkpoint      |

### Configuration Files
- [src_sgl/ucf_option.py](src_sgl/ucf_option.py): UCF-Crime dataset options (paths, epochs, LR, etc.)
- [src_sgl/xd_option.py](src_sgl/xd_option.py): XD-Violence dataset options (paths, epochs, LR, etc.)

## Acknowledgments
We thank the authors of the original VadCLIP model and the creators of the UCF-Crime and XD-Violence datasets for making their resources publicly available.
