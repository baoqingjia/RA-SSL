import os
import torch
import argparse
import time
import datetime
import sys
import scipy.io as scio
from network import Network_3D_Unet
import numpy as np
from utils import FFTKSpace2XSpace_numpy, read_yaml, plot_cpu
from data_process import get_train_no_ksp_logname, test_preprocess_lessMemoryNoTail
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=30, help="number of training epochs")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--GPU', type=int, default=7, help="the index of GPU you will use for computation")
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
parser.add_argument('--data_path', type=str, default='data', help="dataset root path")
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help="checkpoint root path")
parser.add_argument('--test_datasize', type=int, default=3570, help='dataset size to be tested')
parser.add_argument('--center_kspace_size', type=int, default=2, help='central k-space width used for SVD')
parser.add_argument('--rank', type=int, default=8, help='low-rank spectral basis rank')
opt = parser.parse_args()

print('Test parameters:')
print(opt)

model_path = opt.checkpoint_path
model_list = list(os.walk(model_path, topdown=False))[-1][-1]

for i in range(len(model_list)):
    aaa = model_list[i]
    if '.yaml' in aaa:
        yaml_name = model_list[i]
print(f"Using parameter file: {yaml_name}")

read_yaml(opt, os.path.join(model_path, yaml_name))

no_ksp_iz12, no_ksp_all = test_preprocess_lessMemoryNoTail(opt)


denoise_generator = Network_3D_Unet(in_channels = 2,
                                out_channels = 2,
                                final_sigmoid = True)

USE_CUDA = torch.cuda.is_available()
device = torch.device(f"cuda:{opt.GPU}" if USE_CUDA else "cpu")

dataname = get_train_no_ksp_logname(opt.data_path)
test_result_path = os.path.join('result', 'test')
os.makedirs(test_result_path, exist_ok=True)
log_dir = 'test_logs_' + dataname
writer = SummaryWriter(log_dir=log_dir)

n = 1

for checkpoint_index in range(len(model_list)):
    aaa = model_list[checkpoint_index]
    if '.pth' in aaa:
        checkpoint_name = model_list[checkpoint_index]

        denoise_generator.to(device)

        denoise_generator.load_state_dict(torch.load(os.path.join(model_path, checkpoint_name)))

        prev_time = time.time()
        time_start=time.time()

        denoise_img = None
        no_lr_pre_re_im_iz12 = None
        num_test_indices = min(2, no_ksp_iz12.shape[0])

        for index in range(num_test_indices):
            
            no_ksp_index = no_ksp_iz12[index, ...] # (t, w, h, s)
            # Reorder axes to (W, H, S, T).
            no_ksp_trans = no_ksp_index.transpose(1,2,3,0) # (w, h, s, t)
            no_ksp_all_trans = no_ksp_all # (w, h, s, t)

            # Transform k-space to image domain.
            no_ksp_fft = np.fft.fftshift(np.fft.fft(no_ksp_trans, axis=2), axes=2)
            no_ksp_fft = FFTKSpace2XSpace_numpy(no_ksp_fft, 0)
            no = FFTKSpace2XSpace_numpy(no_ksp_fft, 1)

            no_ksp_all_fft = np.fft.fftshift(np.fft.fft(no_ksp_all_trans, axis=2), axes=2)
            no_ksp_all_fft = FFTKSpace2XSpace_numpy(no_ksp_all_fft, 0)
            no_all = FFTKSpace2XSpace_numpy(no_ksp_all_fft, 1)

            # Use the central spatial region for SVD.
            w, h, s, t = no.shape

            reduce_size = (w - opt.center_kspace_size) // 2
            no_raw = no_ksp_trans[reduce_size:-reduce_size,reduce_size:-reduce_size]
            no_raw = no_raw.reshape((w - reduce_size * 2) * (w - reduce_size * 2)* s, -1) # (480, 32)

            no_all_raw = no_ksp_all_trans[reduce_size:-reduce_size,reduce_size:-reduce_size]
            no_all_raw = no_all_raw.reshape((w - reduce_size * 2) * (w - reduce_size * 2)* s, -1) # (480, 32)

            # Estimate low-rank spectral bases.
            rank = opt.rank

            u_p, s_p, vt_p = np.linalg.svd(no_raw, full_matrices=False) # (480, 32) (32,) (32, 32)
            phi = vt_p[:rank,] # (rank, 32)

            u_all, s_all, vt_all = np.linalg.svd(no_all_raw, full_matrices=False) # (480, 32) (32,) (32, 32)
            phi_all = vt_all[:rank,] # (rank, 32)

            no_reshape = no.reshape(w * h * s, t)
            space_coefficient = np.matmul(no_reshape, phi.T.conj()) # (122880, 16)

            # Split complex values into real and imaginary channels.
            space_coefficient_real = np.real(space_coefficient).astype(np.float32)
            space_coefficient_imag = np.imag(space_coefficient).astype(np.float32)
            space_coefficient_comp = np.stack((space_coefficient_real, space_coefficient_imag), axis=-1) # (64, 64, 120, 8, 2)
            space_coefficient = space_coefficient_real + 1j * space_coefficient_imag

            phi_real = np.real(phi).astype(np.float32)
            phi_imag = np.imag(phi).astype(np.float32)
            phi = phi_real + 1j * phi_imag

            phi_all_real = np.real(phi_all).astype(np.float32)
            phi_all_imag = np.imag(phi_all).astype(np.float32)
            phi_all = phi_all_real + 1j * phi_all_imag

            # Reshape tensors.
            space_coefficient_resha = space_coefficient_comp.reshape(w * h, s, rank, -1) # (4096, 120, 8, 2)

            # Build the network input tensor.
            space_coefficient_input = torch.from_numpy(np.expand_dims(space_coefficient_resha, 0)) # (1, 4096, 120, 8, 2)
            space_coefficient_input = space_coefficient_input.permute([0, 4, 3, 1, 2]) # (1, 2, 8, 4096, 120)
            space_coefficient_input = space_coefficient_input.to(device)

            # Move tensors to the selected device.
            phi = torch.from_numpy(phi).to(device) # (16, 32)
            phi_all = torch.from_numpy(phi_all).to(device)

            with torch.no_grad():
                space_coefficient_pre = denoise_generator(space_coefficient_input) # (1, 2, 8, 4096, 120)

            # Remove singleton dimensions.
            space_coefficient_pre = space_coefficient_pre.squeeze()

            # Reorder dimensions for complex reconstruction.
            space_coefficient_pre = space_coefficient_pre.permute([0, 2, 3, 1]) # (2, 1024, 120, 16)

            # Recombine real and imaginary channels.
            space_coefficient_pre = space_coefficient_pre[0, :, :, :] + 1j * space_coefficient_pre[1, :, :, :] # (4096, 120, 8)
            space_coefficient_pre = space_coefficient_pre.reshape(w, h, s, -1)
            space_coefficient_pre = space_coefficient_pre.reshape(w * h * s, -1) # (122880, 8)

            # Reconstruct low-rank images.
            no_lr_pre = torch.matmul(space_coefficient_pre, phi_all) # (122880, 8) (8, 32) > (122880, 32)

            # Reshape tensors.
            no_lr_pre = no_lr_pre.reshape(w, h, s, t) # (w, h, s, t)

            no_lr_pre_real = no_lr_pre.real
            no_lr_pre_imag = no_lr_pre.imag
            no_lr_pre_re_im = torch.stack((no_lr_pre_real, no_lr_pre_imag), dim=0) # (2, w, h, s, t)

            if no_lr_pre_re_im_iz12 is None:
                no_lr_pre_re_im_iz12 = np.zeros((num_test_indices, 2, w, h, s, t), dtype=np.float32)

            no_lr_pre_re_im_iz12[index, ...] = no_lr_pre_re_im.detach().cpu().numpy()
            no_lr_pre_re_im_iz12_mean = np.mean(no_lr_pre_re_im_iz12[:index + 1], axis=0)
            denoise_img = no_lr_pre_re_im_iz12_mean[0, ...] + 1j * no_lr_pre_re_im_iz12_mean[1, ...]

            # Determine approximate time left
            batches_done = index
            batches_left = num_test_indices - batches_done - 1
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            prev_time = time.time()

            if index%1 == 0:
                time_end=time.time()
                time_cost=datetime.timedelta(seconds= (time_end - time_start))
                sys.stdout.write("\r [Batch %d/%d] [Time Left: %s] [Time Cost: %s]"
                % (index,
                num_test_indices,
                time_left,
                time_cost,))

            fig_pre = plot_cpu(no_lr_pre.detach().cpu().numpy())
            writer.add_figure(f'1_de', fig_pre, n)
            plt.close(fig_pre)    

            fig_no_all = plot_cpu(no_all)
            writer.add_figure(f'2_no_all', fig_no_all, n)
            plt.close(fig_no_all)   

            fig_no = plot_cpu(no)
            writer.add_figure(f'3_no', fig_no, n)
            plt.close(fig_no) 

            n += 1 

        if denoise_img is None:
            raise RuntimeError('No test output was generated.')

        de = os.path.join(test_result_path, dataname + '.mat')
        scio.savemat(de, {'de': denoise_img})

        print("Test reconstruction saved.")

writer.close()