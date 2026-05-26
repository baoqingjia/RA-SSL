import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import random


def FFTKSpace2XSpace_numpy(PreFFT, Dim):
    # Perform inverse FFT shift, then FFT, and shift again
    PostFFT = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(PreFFT, axes=Dim), axis=Dim), axes=Dim)
    return PostFFT


def FFTXSpace2KSpace_numpy(PreFFT, Dim):
    PostFFT = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(PreFFT, axes=Dim), axis=Dim), axes=Dim)
    return PostFFT


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


def save_yaml(opt, yaml_name):
    para = {
        "epoch": opt.epoch,
        "n_epochs": opt.n_epochs,
        "batch_size": opt.batch_size,
        "lr": opt.lr,
        "b1": opt.b1,
        "b2": opt.b2,
        "data_path": opt.data_path,
        "center_kspace_size": opt.center_kspace_size,
        "rank": opt.rank,
    }
    with open(yaml_name, 'w') as f:
        yaml.dump(para, f)


def read_yaml(opt, yaml_name):
    with open(yaml_name) as f:
        para = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded parameters: {para}")
        opt.epoch = para["epoch"]
        opt.n_epochs = para["n_epochs"]
        opt.batch_size = para["batch_size"]
        opt.lr = para["lr"]
        opt.b1 = para["b1"]
        opt.b2 = para["b2"]
        opt.center_kspace_size = para.get("center_kspace_size", opt.center_kspace_size)
        opt.rank = para.get("rank", opt.rank)


def plot_cpu(data):
    x, y, z, t = data.shape

    data_re = abs(data) / abs(data).max()

    data_re = np.transpose(data_re, (0,1,2,3))

    data_reshape = data_re.reshape(x * y, z, t)

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(3, 7, width_ratios=[1.5, 1, 1.5, 1, 1.5, 1, 5])


    i_x = 18
    i_y = 15

    met = [67, 61, 49]

    water_dy = data_re[i_x, i_y,met[0],:]
    glu_dy = data_re[i_x, i_y,met[1],:]
    lac_dy = data_re[i_x, i_y,met[2],:]
    x_length = np.arange(32)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    ax6 = fig.add_subplot(gs[5])
    ax7 = fig.add_subplot(gs[1:3, 0:4])

    # lac
    dy = 15
    met_n = 49
    data_trans = data_reshape.transpose(1, 0, 2)[..., dy]  # (120, 1024)
    ax1.plot(data_trans)
    ax1.set_title(f'dy: {dy}, met_n: {met_n}')
    ax1.axis('off')

    met = data_re[:, :, met_n, dy]
    ax2.imshow(met, cmap='jet')
    ax2.set_title(f'dy: {dy}, met_n: {met_n}')
    ax2.axis('off')

    # glu
    dy = 7
    met_n = 61
    data_trans = data_reshape.transpose(1, 0, 2)[..., dy]  # (120, 1024)
    ax3.plot(data_trans)
    ax3.set_title(f'dy: {dy}, met_n: {met_n}')
    ax3.axis('off')

    met = data_re[:, :, met_n, dy]
    ax4.imshow(met, cmap='jet')
    ax4.set_title(f'dy: {dy}, met_n: {met_n}')
    ax4.axis('off')

    # water
    dy = 23
    met_n = 67
    data_trans = data_reshape.transpose(1, 0, 2)[..., dy]  # (120, 1024)
    ax5.plot(data_trans)
    ax5.set_title(f'dy: {dy}, met_n: {met_n}')
    ax5.axis('off')

    met = data_re[:, :, met_n, dy]
    ax6.imshow(met, cmap='jet')
    ax6.set_title(f'dy: {dy}, met_n: {met_n}')
    ax6.axis('off')

    ax7.plot(x_length, water_dy, label='water', color='k', linestyle='-', marker='o')
    ax7.plot(x_length, glu_dy, label='glu', color='b', linestyle='-', marker='o')
    ax7.plot(x_length, lac_dy, label='lac', color='r', linestyle='-', marker='o')
    ax7.set_title('dy', fontsize=14)
    ax7.set_xlabel('times', fontsize=12)
    ax7.set_ylabel('Values', fontsize=12)
    ax7.legend()

    fig.tight_layout()

    return fig


class DMIDataAugmentorN2N:
    def __init__(self):
        pass

    def __call__(self, x1, x2, x3, x4):
        assert x1.shape == x2.shape == x3.shape, "All inputs must have the same shape"
        # Reuse one random seed so each paired input receives the same augmentation.
        seed = random.randint(0, 999999)

        random.seed(seed)
        x1_aug = self.augment_single(x1)

        random.seed(seed)
        x2_aug = self.augment_single(x2)

        random.seed(seed)
        x3_aug = self.augment_single(x3)

        random.seed(seed)
        x4_aug = self.augment_single(x4)

        return x1_aug, x2_aug, x3_aug, x4_aug

    def augment_single(self, x):
        # Random flips.
        if random.random() < 0.5:
            x = np.flip(x, axis=0)  # W direction
        if random.random() < 0.5:
            x = np.flip(x, axis=1)  # H direction

        # Random rotation by a multiple of 90 degrees.
        k = random.choice([0, 1, 2, 3])
        x = np.rot90(x, k=k, axes=(0, 1))  # Rotate in the spatial (W, H) plane.

        return x