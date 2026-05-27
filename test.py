import os
import torch
import argparse
import time
import datetime
import sys
import shutil
import scipy.io as scio
from network import SSAN_Net
import numpy as np
from utils import FFTKSpace2XSpace_numpy, read_yaml, plot_cpu
from data_process import test_data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--GPU', type=int, default=6, help="the index of GPU you will use for computation")
parser.add_argument('--data_path', type=str, default=os.path.join('dataset', 'dmi_si_hum32_no008_ra32'), help="dataset path")
parser.add_argument('--test_ksp_name', type=str, default=None, help="test k-space .mat file name; default uses accumulated no_ksp all file when available")
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help="checkpoint root path")
parser.add_argument('--model_name', type=str, default='epoch_8.pth', help="checkpoint model name to load, e.g. epoch_30.pth")
parser.add_argument('--center_kspace_size', type=int, default=2, help='central k-space width used for SVD')
parser.add_argument('--rank', type=int, default=8, help='low-rank spectral basis rank')
opt = parser.parse_args()

print('Test parameters:')
print(opt)

data_group_name = os.path.basename(os.path.normpath(opt.data_path))
model_path = os.path.join(opt.checkpoint_path, data_group_name)
if not os.path.isdir(model_path):
    raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")
model_list = sorted(os.listdir(model_path))

yaml_names = [name for name in model_list if name.endswith('.yaml')]
if len(yaml_names) == 0:
    raise FileNotFoundError(f"No .yaml parameter file found in {model_path}.")
yaml_name = yaml_names[0]
print(f"Using parameter file: {yaml_name}")

read_yaml(opt, os.path.join(model_path, yaml_name))

if opt.model_name is not None:
    checkpoint_name = opt.model_name
    if not checkpoint_name.endswith('.pth'):
        checkpoint_name = checkpoint_name + '.pth'
    checkpoint_names = [checkpoint_name]
else:
    checkpoint_names = [name for name in model_list if name.endswith('.pth')]

if len(checkpoint_names) == 0:
    raise FileNotFoundError(f"No .pth checkpoint found in {model_path}.")

for checkpoint_name in checkpoint_names:
    checkpoint_file = os.path.join(model_path, checkpoint_name)
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

model_list = checkpoint_names

no_ksp_test, no_ksp_all, gt_ksp, test_no_ksp_path = test_data(opt)
if no_ksp_test.ndim not in (4, 5):
    raise ValueError(f"Expected 4D or 5D test k-space, got shape {no_ksp_test.shape}.")

test_has_group = no_ksp_test.ndim == 5
if test_has_group:
    print(f"Using grouped test k-space: {test_no_ksp_path}; averaging first {min(2, no_ksp_test.shape[0])} groups.")
else:
    print(f"Using accumulated 4D test k-space: {test_no_ksp_path}; averaging is skipped.")

has_gt = gt_ksp is not None

if has_gt:
    gt_ksp_fft = np.fft.fftshift(np.fft.fft(gt_ksp, axis=2), axes=2)
    gt_ksp_fft = FFTKSpace2XSpace_numpy(gt_ksp_fft, 0)
    gt = FFTKSpace2XSpace_numpy(gt_ksp_fft, 1)
else:
    gt = None
    print("No gt_ksp found. RMSE and GT TensorBoard visualization will be skipped.")


denoise_generator = SSAN_Net(in_channels = 2,
                                out_channels = 2,
                                final_sigmoid = True)

USE_CUDA = torch.cuda.is_available()
device = torch.device(f"cuda:{opt.GPU}" if USE_CUDA else "cpu")

logname = os.path.splitext(os.path.basename(test_no_ksp_path))[0]
test_result_path = os.path.join('result', 'test', data_group_name)
os.makedirs(test_result_path, exist_ok=True)
log_dir = os.path.join('log', 'test', data_group_name)

if os.path.exists(log_dir) and os.listdir(log_dir):
    print(f"Directory {log_dir} is not empty. It will be cleared.")
    shutil.rmtree(log_dir)
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

n = 1

for checkpoint_index in range(len(model_list)):
    aaa = model_list[checkpoint_index]
    if '.pth' in aaa:
        checkpoint_name = model_list[checkpoint_index]
        print(f"Using checkpoint file: {checkpoint_name}")

        denoise_generator.to(device)

        denoise_generator.load_state_dict(torch.load(os.path.join(model_path, checkpoint_name)))

        prev_time = time.time()
        time_start=time.time()

        denoise_img = None
        no_lr_pre_re_im_iz12 = None
        num_test_indices = min(2, no_ksp_test.shape[0]) if test_has_group else 1

        for index in range(num_test_indices):
            
            if test_has_group:
                no_ksp_index = no_ksp_test[index, ...] # (t, w, h, s)
                # Reorder axes to (W, H, S, T).
                no_ksp_trans = no_ksp_index.transpose(1,2,3,0) # (w, h, s, t)
            else:
                no_ksp_trans = no_ksp_test # (w, h, s, t)
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

            # Build the SSAN_Net input tensor.
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

            if test_has_group:
                if no_lr_pre_re_im_iz12 is None:
                    no_lr_pre_re_im_iz12 = np.zeros((num_test_indices, 2, w, h, s, t), dtype=np.float32)

                no_lr_pre_re_im_iz12[index, ...] = no_lr_pre_re_im.detach().cpu().numpy()
                no_lr_pre_re_im_iz12_mean = np.mean(no_lr_pre_re_im_iz12[:index + 1], axis=0)
                denoise_img = no_lr_pre_re_im_iz12_mean[0, ...] + 1j * no_lr_pre_re_im_iz12_mean[1, ...]
            else:
                denoise_img = no_lr_pre.detach().cpu().numpy()

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

            if has_gt:
                fig_gt = plot_cpu(gt)
                writer.add_figure(f'2_gt', fig_gt, n)
                plt.close(fig_gt)  
                no_all_tag = '3_no_all'
                no_tag = '4_no'
            else:
                no_all_tag = '2_no_all'
                no_tag = '3_no'

            fig_no_all = plot_cpu(no_all)
            writer.add_figure(no_all_tag, fig_no_all, n)
            plt.close(fig_no_all)   

            fig_no = plot_cpu(no)
            writer.add_figure(no_tag, fig_no, n)
            plt.close(fig_no) 

            n += 1 

        if denoise_img is None:
            raise RuntimeError('No test output was generated.')

        de = os.path.join(test_result_path, logname + '.mat')
        scio.savemat(de, {'de': denoise_img})

        if has_gt:
            gt_abs_norm = abs(gt) / abs(gt).max()
            denoise_img_abs_norm = abs(denoise_img) / abs(denoise_img).max()
            rmse_iz12 = np.sqrt(np.mean((denoise_img_abs_norm[:,:,49:68,:] - gt_abs_norm[:,:,49:68,:])**2))
            rmse_text = f"rmse_iz12: {rmse_iz12:.6f}"

            if test_has_group and no_lr_pre_re_im_iz12 is not None and no_lr_pre_re_im_iz12.shape[0] >= 2:
                iz1_abs_norm = abs(no_lr_pre_re_im_iz12[0,...]) / abs(no_lr_pre_re_im_iz12[0,...]).max()
                iz2_abs_norm = abs(no_lr_pre_re_im_iz12[1,...]) / abs(no_lr_pre_re_im_iz12[1,...]).max()
                rmse_iz1 = np.sqrt(np.mean((iz1_abs_norm[0,:,:,49:68,:] - gt_abs_norm[:,:,49:68,:])**2))
                rmse_iz2 = np.sqrt(np.mean((iz2_abs_norm[1,:,:,49:68,:] - gt_abs_norm[:,:,49:68,:])**2))
                rmse_text += f" rmse_iz1: {rmse_iz1:.6f} rmse_iz2: {rmse_iz2:.6f}"

            print(rmse_text)

        print("Test reconstruction saved.")

writer.close()