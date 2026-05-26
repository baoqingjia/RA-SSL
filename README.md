# RA-SSL

This repository contains the training and inference code for an RA-SSL reconstruction pipeline built around a spectral-spatial attention network (`SSAN`). The code is organized for dynamic metabolic imaging k-space data stored in MATLAB/HDF5 `.mat` files.

## Overview

The pipeline trains a self-supervised reconstruction network from paired noisy k-space measurements. Each sample is transformed into the image domain, projected onto low-rank spectral bases, processed by the SSAN model, and reconstructed back into complex-valued image sequences.

Main features:

- Self-supervised training from noisy k-space pairs.
- Low-rank spectral representation with configurable rank.
- Configurable central k-space region for SVD basis estimation.
- Checkpoint saving by epoch.
- Separate output folders for training and testing reconstructions.

## Repository Structure

```text
RA-SSL/
├── SSAN.py              # Spectral-spatial attention network
├── buildingblocks.py    # 3D U-Net encoder/decoder building blocks
├── data_process.py      # Data discovery and preprocessing
├── network.py           # Model wrapper
├── train.py             # Training script
├── test.py              # Inference script
├── utils.py             # FFT, YAML, plotting, and augmentation utilities
├── data/                # Input .mat files
├── checkpoint/          # Model checkpoints and training parameters
└── result/
    ├── train/           # Training-time reconstruction outputs
    └── test/            # Test-time reconstruction outputs
```

## Data Preparation

Place the required `.mat` files under `data/`.

The current preprocessing code expects one file for each role:

- Summed/all noisy k-space file: filename contains both `no_ksp` and `all`
- Reference k-space file: filename contains `gt_ksp`

Example:

```text
data/
├── dmi_si_gt_ksp_hum32.mat
└── dmi_si_no_ksp_hum32_no08_ra32_all.mat
```

The training log name is generated automatically from the noisy k-space filename without the `.mat` suffix.

## Environment

The code is written in Python and uses PyTorch. A CUDA-capable GPU is recommended.

Core dependencies:

```text
torch
numpy
scipy
h5py
matplotlib
tensorboard
pyyaml
einops
einops-exts
```

Install dependencies in your preferred environment, for example:

```bash
pip install torch numpy scipy h5py matplotlib tensorboard pyyaml einops einops-exts
```

## Training

Run:

```bash
python train.py --GPU 0
```

Common options:

```bash
python train.py \
  --GPU 0 \
  --n_epochs 30 \
  --lr 0.00005 \
  --center_kspace_size 2 \
  --rank 8
```

Important arguments:

- `--GPU`: GPU index used when CUDA is available.
- `--epoch`: starting epoch. Use values greater than `1` to resume from the previous checkpoint.
- `--n_epochs`: number of training epochs.
- `--data_path`: input data directory. Default: `data`.
- `--checkpoint_path`: checkpoint directory. Default: `checkpoint`.
- `--center_kspace_size`: central k-space width used for SVD basis estimation. Default: `2`.
- `--rank`: low-rank spectral basis rank. Default: `8`.

Training outputs:

```text
checkpoint/
├── para.yaml
├── output.txt
├── epoch_1.pth
├── epoch_2.pth
└── ...

result/train/
└── <logname>_epoch_<epoch>.mat
```

TensorBoard logs are written to:

```text
logs_<logname>/
```

## Testing

Run:

```bash
python test.py --GPU 0
```

The test script loads parameters from `checkpoint/para.yaml` and applies all `.pth` checkpoints found in the checkpoint directory.

Test outputs:

```text
result/test/
└── <logname>.mat
```

TensorBoard logs are written to:

```text
test_logs_<logname>/
```

## Notes

- Checkpoints are saved as PyTorch state dictionaries.
- Reconstruction outputs are saved as MATLAB `.mat` files with the key `de`.
- The test script uses the first two noisy k-space measurements and saves their averaged complex reconstruction.
- Existing TensorBoard log directories with the same name may be cleared by the training script.

## References

1. Lehtinen J, Munkberg J, Hasselgren J, et al.  
   **Noise2Noise: Learning image restoration without clean data**[J].  
   *arXiv preprint arXiv:1803.04189*, 2018.

2. Li X, Zhang G, Wu J, et al.  
   **Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising**[J].  
   *Nature Methods*, 2021, 18(11): 1395–1400.
