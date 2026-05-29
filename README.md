# RA-SSL

This repository contains training and inference code for an RA-SSL reconstruction pipeline built around the spectral-spatial attention network `SSAN`. The code works with dynamic metabolic imaging k-space data stored in MATLAB `.mat` files.

## Project Layout

```text
RA-SSL/
├── SSAN.py
├── buildingblocks.py
├── data_process.py
├── network.py
├── train.py
├── test.py
├── utils.py
├── dataset/
│   └── <dataset_name>/
├── checkpoint/
│   └── <dataset_name>/
├── result/
│   ├── train/
│   │   └── <dataset_name>/
│   └── test/
│       └── <dataset_name>/
└── log/
    ├── train/
    │   └── <dataset_name>/
    └── test/
        └── <dataset_name>/
```

The current dataset directory is:

```text
dataset/dmi_si_hum32_no008_ra32/
```

The dataset name, for example `dmi_si_hum32_no008_ra32`, is used automatically to group checkpoints, results, and TensorBoard logs.

## Data Preparation

Place `.mat` files under one dataset subdirectory. The grouped noisy k-space file `dmi_si_no_ksp_hum32_no008_ra32_addno02.mat` is not uploaded because it is large; please contact the authors by email if you need access to this file:

```text
dataset/dmi_si_hum32_no008_ra32/
├── dmi_si_gt_ksp_hum32.mat
├── dmi_si_no_ksp_hum32_no008_ra32_addno02.mat
└── dmi_si_no_ksp_hum32_no08_all.mat
```

Training expects:

- A grouped noisy k-space file whose name contains `no_ksp` and does not contain `all`.
- An accumulated/all noisy k-space file whose name contains both `no_ksp` and `all`.
- An optional reference file whose name contains `gt_ksp`.

Testing input:

- 4D accumulated/all mode: a file with key `no_ksp`, shape `(w, h, s, t)` after preprocessing.

If `gt_ksp` exists, train/test compute RMSE and write GT images to TensorBoard. If it does not exist, RMSE and GT visualization are skipped.

## Environment

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

Install dependencies in your Python environment, for example:

```bash
pip install torch numpy scipy h5py matplotlib tensorboard pyyaml einops einops-exts
```

## Training

Run training with the dataset directory:

```bash
python train.py \
  --data_path dataset/dmi_si_hum32_no008_ra32 \
  --GPU 0
```

Common options:

```bash
python train.py \
  --data_path dataset/dmi_si_hum32_no008_ra32 \
  --GPU 0 \
  --n_epochs 30 \
  --lr 0.00005 \
  --center_kspace_size 2 \
  --rank 8
```

Important arguments:

- `--data_path`: dataset directory containing the `.mat` files.
- `--checkpoint_path`: checkpoint root directory. The script writes to `checkpoint/<dataset_name>/`.
- `--epoch`: starting epoch. Use values greater than `1` to resume from `epoch_<epoch-1>.pth`; it must be less than or equal to `--n_epochs`.
- `--n_epochs`: final epoch to train through, inclusive. For example, `--epoch 1 --n_epochs 30` trains epochs 1 through 30.
- `--GPU`: GPU index used when CUDA is available.
- `--center_kspace_size`: central k-space width used for SVD basis estimation.
- `--rank`: low-rank spectral basis rank.

Training outputs:

```text
checkpoint/<dataset_name>/
├── para.yaml
├── output.txt
├── epoch_1.pth
├── epoch_2.pth
└── ...

result/train/<dataset_name>/
├── <train_no_ksp_logname>_console.log
├── <train_no_ksp_logname>_epoch_1.mat
├── <train_no_ksp_logname>_epoch_2.mat
└── ...

log/train/<dataset_name>/
└── events.out.tfevents...
```

`train.py` clears a non-empty TensorBoard log directory before creating a new `SummaryWriter`.

## Testing

By default, `test.py` uses the accumulated/all 4D file whose name contains both `no_ksp` and `all`:

```bash
python test.py \
  --data_path dataset/dmi_si_hum32_no008_ra32 \
  --model_name epoch_8.pth \
  --GPU 0
```

This default all-file mode skips group averaging.

To test a grouped `no_n2n_ksp` file and average the first two groups, pass the file name explicitly:

```bash
python test.py \
  --data_path dataset/dmi_si_hum32_no008_ra32 \
  --test_ksp_name dmi_si_no_ksp_hum32_no008_ra32_addno02.mat \
  --model_name epoch_8.pth \
  --GPU 0
```

Important arguments:

- `--data_path`: dataset directory containing the `.mat` files.
- `--test_ksp_name`: optional test `.mat` file name. If omitted, the script uses the accumulated/all `no_ksp` file when available.
- `--checkpoint_path`: checkpoint root directory. The script reads from `checkpoint/<dataset_name>/`.
- `--model_name`: checkpoint file to load. The `.pth` suffix is optional.
- `--GPU`: GPU index used when CUDA is available.

Testing outputs:

```text
result/test/<dataset_name>/
└── <test_ksp_file_name_without_ext>.mat

log/test/<dataset_name>/
└── events.out.tfevents...
```

`test.py` clears a non-empty TensorBoard log directory before creating a new `SummaryWriter`. Because both all-file and group-file tests currently write TensorBoard events to `log/test/<dataset_name>/`, the later test run replaces the previous TensorBoard log. The saved `.mat` results are separated by test file name.

## Notes

- Reconstruction outputs are MATLAB `.mat` files with the key `de`.
- Checkpoints are PyTorch state dictionaries.
- Training pairs are generated from the first dimension of the preprocessed `no_n2n_ksp`, so the number of grouped noisy measurements is not hard-coded.
- `para.yaml` is saved with training hyperparameters and loaded by `test.py` before reconstruction. The dataset path used for grouping checkpoints/results still comes from `--data_path`.

## References

1. Lehtinen J, Munkberg J, Hasselgren J, et al. Noise2Noise: Learning image restoration without clean data[J]. arXiv preprint arXiv:1803.04189, 2018.

2. Li X, Zhang G, Wu J, et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising[J]. Nature Methods, 2021, 18(11): 1395-1400.
