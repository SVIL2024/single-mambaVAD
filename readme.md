# MambaVAD

Official PyTorch implementation of the manuscript

> **MambaVAD: A Shared-Parameter Multi-Scale Mamba Network for Efficient Weakly Supervised Video Anomaly Detection**

The model is a single Mamba2 encoder over frozen CLIP ViT-B/16 features. It can scan the
sequence in both directions (or forward-only for streaming use), fuses several temporal
resolutions through **one** set of parameters, and is trained with a classifier loss plus
two auxiliary terms: a normal-prototype contrastive loss and a score-consistency
regulariser.

## What is in this repository

| Path | Content |
| --- | --- |
| `src/` | The pipeline behind the reported numbers: model, losses, training/testing, and the experiment driver that produced every table in the paper. |
| `src_sgl/` | Earlier single-encoder release of the same model (one temporal scale, no experiment driver). Kept for reference; the results in the paper come from `src/`. |
| `list/` | Video lists (`ucf_CLIP_*.csv`, `xd_CLIP_*.csv`), ground truth and the scripts that regenerate them. |
| `results/` | The JSON audit trail of the runs quoted in the paper (one file per run, plus per-grid summaries). |

## Method in one paragraph

`CLIP features -> position embedding -> bidirectional Mamba2 -> shared-parameter
multi-scale fusion -> linear scoring head`. The classifier is a top-k MIL head. The
prototype term pulls normal video representations towards a visual prototype estimated
from the normal videos of the batch, and pushes anomalous ones away; its repulsive half
can act on the whole video mean (`--proto-pool mean`) or on the highest-scoring frames
(`--proto-pool topk`, `--proto-pool soft`). The consistency term matches the score of a
frame with the model's own score at an offset, `s_{t+offset}`, and is a smoothing
regulariser on the scoring function rather than a forecast.

## Installation

```bash
pip install torch torchvision            # CUDA build matching your driver
pip install mamba-ssm causal-conv1d --no-build-isolation
pip install numpy pandas scikit-learn
```

`causal-conv1d` and the selective-scan CUDA extension are required: without them
`mamba_ssm` falls back to a slow path or fails on `cuda`.

## Data

Pre-extracted CLIP features for UCF-Crime and XD-Violence are used throughout (the paper
therefore isolates the temporal model from the feature extractor):

| Benchmark | CLIP [Baidu] | CLIP [OneDrive] |
| --- | --- | --- |
| UCF-Crime | [Code: 7yzp](https://pan.baidu.com/s/1OKRIxoLcxt-7RYxWpylgLQ) | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Ea86YOcp5z9KhRFDQm9a8zwBcGiGGg5BuBJtgmCVByazBQ?e=tqHLHt) |
| XD-Violence | [Code: v8tw](https://pan.baidu.com/s/1q8DiYHcPJtrBQiiJMI7aJw) | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/Et5dWQZb2cBDs7zsrp90SrQBL_52vTRNYTdjQW6SMl0ZVA?e=foX4ph) |

Each video is stored as ten crops (`...__0.npy` ... `...__9.npy`), 256 segments x 512
dimensions, `float16`. After extracting, rewrite the path prefix of the four lists
(`list/ucf_CLIP_rgb.csv`, `list/ucf_CLIP_rgbtest.csv`, `list/xd_CLIP_rgb.csv`,
`list/xd_CLIP_rgbtest.csv`) to point at your local copy.

## Reproducing the paper

All commands are run from `src/`. The protocol used for the main results is 10 epochs,
batch 32, `lr 1e-5`, dropout 0.5, prediction stride 1.

```bash
cd src

# Table 1 / abstract: main configuration, three seeds
python -m experiment.run_grid --step g1 --seeds 234 789 999 \
    --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1

# Table 3: forward-only scan, multi-scale sweep, loss components,
# the offset/pool study and the alignment control
python -m experiment.run_grid --step g2 --seeds 234 789 999 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
python -m experiment.run_grid --step g3 --seeds 789 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
python -m experiment.run_grid --step g4 --seeds 789 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
python -m experiment.run_grid --step g5 --seeds 789 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
python -m experiment.run_grid --step g6 --seeds 789 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
python -m experiment.run_grid --step g7 --seeds 789 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
python -m experiment.run_grid --step g8 --seeds 789 --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1
```

A single configuration can also be trained directly:

```bash
python -m experiment.run_one --exp full --epochs 10 --batch 32 --lr 1e-5 --dropout 0.5 --pred-stride 1 --seed 789
```

Auxiliary-term switches (defaults reproduce the reported configuration):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--proto-pool` | `mean` | what the repulsive half of the prototype loss acts on: `mean`, `topk`, `soft` |
| `--proto-topk-ratio` | `0.1` | fraction of frames kept by `topk` (10 % of the video length) |
| `--proto-tau` | `0.1` | softmax temperature for `soft` |
| `--future-align` | `legacy` | `legacy` = the exact padded self-consistency used for every number in the paper (offset 0); `shift` = padded-free offset, to be combined with `--future-horizon` |
| `--future-horizon` | `1` | offset `k` of `--future-align shift`; `k = 0` gives a padded-free self-consistency |

Self-check of the auxiliary losses (no GPU needed):

```bash
cd src && python -m experiment.selftest_realfix     # 18 checks
```

Latency and memory (single stream, batch 1):

```bash
cd src && python -m experiment.latency_infer --stream
```

## Notes on the reported numbers

* Every run in `results/` is annotated with the seed, epoch budget and report file. Rows
  quoted from other papers are marked `n/r` when the original authors did not report that
  metric on that dataset.
* The training-step benchmark of the paper is measured on an RTX 2080 Ti with the fast
  `mamba_ssm` path; the single-stream inference table uses an RTX 3060 and is therefore
  not numerically comparable to it. Both are stated in the paper.
* Checkpoints are not committed; they are available on request.

## Acknowledgement

The CLIP wrapper, the feature lists and the multiple-instance learning scaffolding are
adapted from the public VadCLIP release (Wu et al., AAAI 2024), and the datasets come from
the UCF-Crime and XD-Violence authors. We thank them for making these resources public.
