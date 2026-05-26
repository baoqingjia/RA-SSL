import os
import torch
import argparse
import time
import sys
from network import Network_3D_Unet
import numpy as np
from data_process import get_train_no_ksp_logname, train_preprocess_lessMemoryMulStacks
from utils import FFTKSpace2XSpace_numpy, save_yaml, plot_cpu, DMIDataAugmentorN2N, FFTXSpace2KSpace_numpy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil
import scipy.io as scio


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=30, help="number of training epochs")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--GPU', type=int, default=5, help="the index of GPU you will use for computation")
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
parser.add_argument('--data_path', type=str, default='data', help="dataset root path")
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help="checkpoint root path")
parser.add_argument('--center_kspace_size', type=int, default=2, help='central k-space width used for SVD')
parser.add_argument('--rank', type=int, default=8, help='low-rank spectral basis rank')
opt = parser.parse_args()

print('Training parameters:')
print(opt)

logname = get_train_no_ksp_logname(opt.data_path)

checkpoint_path = opt.checkpoint_path
os.makedirs(checkpoint_path, exist_ok=True)

yaml_name = os.path.join(checkpoint_path, 'para.yaml')
save_yaml(opt, yaml_name)

USE_CUDA = torch.cuda.is_available()
device = torch.device(f"cuda:{opt.GPU}" if USE_CUDA else "cpu")


L2_pixelwise = torch.nn.MSELoss()


denoise_generator = Network_3D_Unet(in_channels = 2,
                                out_channels = 2,
                                final_sigmoid = True)

denoise_generator.to(device)

# Resume training from the previous checkpoint when requested.
if opt.epoch > 1:
    # Load the checkpoint from the previous epoch.
    resume_epoch = opt.epoch - 1
    resume_checkpoint_path = os.path.join(checkpoint_path, 'epoch_' + str(resume_epoch) + '.pth')
    
    if os.path.exists(resume_checkpoint_path):
        print(f"---------> Resuming training from epoch {resume_epoch}...")
        print(f"---------> Loading checkpoint: {resume_checkpoint_path}")
        denoise_generator.load_state_dict(torch.load(resume_checkpoint_path, map_location=device))
        print("---------> Checkpoint loaded successfully!")
    else:
        print(f"---------> Warning: Checkpoint {resume_checkpoint_path} not found!")
        print("---------> Starting from scratch or please check the --epoch parameter.")

coordinate_list, no_ksp, no_ksp_ori, gt_ksp = train_preprocess_lessMemoryMulStacks(opt) # (64, 15, 32, 32, 120)


no_ksp_all_fft = np.fft.fftshift(np.fft.fft(no_ksp_ori, axis=2), axes=2)
no_ksp_all_fft = FFTKSpace2XSpace_numpy(no_ksp_all_fft, 0)
no_all_ori = FFTKSpace2XSpace_numpy(no_ksp_all_fft, 1)

gt_ksp_fft = np.fft.fftshift(np.fft.fft(gt_ksp, axis=2), axes=2)
gt_ksp_fft = FFTKSpace2XSpace_numpy(gt_ksp_fft, 0)
gt = FFTKSpace2XSpace_numpy(gt_ksp_fft, 1)

L2_pixelwise.to(device)
optimizer_G = torch.optim.Adam(denoise_generator.parameters(),
                                lr=opt.lr, betas=(opt.b1, opt.b2))

train_time_sum = 0
epoch_time = 0
time_start=time.time()

log_dir = 'logs_' + logname
train_result_path = os.path.join('result', 'train')
os.makedirs(train_result_path, exist_ok=True)

if os.path.exists(log_dir) and os.listdir(log_dir):
    print(f"Directory {log_dir} is not empty. It will be cleared.")
    shutil.rmtree(log_dir)
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# Reset the epoch metric log.
file_path = os.path.join(checkpoint_path, 'output.txt')

if os.path.exists(file_path):
    os.remove(file_path)

n = 1

for epoch in range(opt.epoch, opt.n_epochs):

    denoise_img = np.zeros(no_all_ori.shape, dtype=complex) # (w, h, s, t)
    w, h, s, t = no_all_ori.shape
    no_p1_lr_pre_re_im_iz12 = np.zeros((2, 2, w, h, s, t))
    no_p1_lr_pre_re_im_iz12_mean = np.zeros((2, w, h, s, t))

    for index in range(len(coordinate_list)):
        
        no_p1_ksp = no_ksp[coordinate_list[index][0], :, :, :]
        no_p2_ksp = no_ksp[coordinate_list[index][1], :, :, :] # (t, w, h, s)

        no_all = no_all_ori
        no_ksp_all = no_ksp_ori

        # Reorder axes to (W, H, S, T).
        no_p1_ksp_trans = no_p1_ksp.transpose(1,2,3,0)
        no_p2_ksp_trans = no_p2_ksp.transpose(1,2,3,0)

        # Transform k-space to image domain.
        no_p1_ksp = np.fft.fftshift(np.fft.fft(no_p1_ksp_trans, axis=2), axes=2)
        no_p1_ksp = FFTKSpace2XSpace_numpy(no_p1_ksp, 0)
        no_p1 = FFTKSpace2XSpace_numpy(no_p1_ksp, 1)

        no_p2_ksp = np.fft.fftshift(np.fft.fft(no_p2_ksp_trans, axis=2), axes=2)
        no_p2_ksp = FFTKSpace2XSpace_numpy(no_p2_ksp, 0)
        no_p2 = FFTKSpace2XSpace_numpy(no_p2_ksp, 1)

        if index > 1:
            augmentor = DMIDataAugmentorN2N()

            no_p1, no_p2, no_all, _ = augmentor(no_p1, no_p2, no_all, no_all)

            no_p1_fft = np.fft.ifft(np.fft.ifftshift(no_p1, axes=2), axis=2)
            no_p1_fft = FFTXSpace2KSpace_numpy(no_p1_fft, 0)
            no_p1_ksp_trans = FFTXSpace2KSpace_numpy(no_p1_fft, 1)

            no_p2_fft = np.fft.ifft(np.fft.ifftshift(no_p2, axes=2), axis=2)
            no_p2_fft = FFTXSpace2KSpace_numpy(no_p2_fft, 0)
            no_p2_ksp_trans = FFTXSpace2KSpace_numpy(no_p2_fft, 1)

            no_all_fft = np.fft.ifft(np.fft.ifftshift(no_all, axes=2), axis=2)
            no_all_fft = FFTXSpace2KSpace_numpy(no_all_fft, 0)
            no_ksp_all = FFTXSpace2KSpace_numpy(no_all_fft, 1)
        
        w, h, s, t = no_p1_ksp_trans.shape

        # Use the central spatial region for SVD.
        reduce_size = (w - opt.center_kspace_size) // 2
        no_p1_raw = no_p1_ksp_trans[reduce_size:-reduce_size,reduce_size:-reduce_size]
        no_p1_raw = no_p1_raw.reshape((w - reduce_size * 2) * (w - reduce_size * 2)* s, -1)

        no_p2_raw = no_p2_ksp_trans[reduce_size:-reduce_size,reduce_size:-reduce_size]
        no_p2_raw = no_p2_raw.reshape((w - reduce_size * 2) * (w - reduce_size * 2) * s, -1)

        no_all_raw = no_ksp_all[reduce_size:-reduce_size,reduce_size:-reduce_size] # (2, 2, 120, 32)
        no_all_raw = no_all_raw.reshape((w - reduce_size * 2) * (w - reduce_size * 2) * s, -1) # (480, 32)

        # Estimate low-rank spectral bases.
        rank = opt.rank

        u_p1, s_p1, vt_p1 = np.linalg.svd(no_p1_raw, full_matrices=False)
        phi_p1 = vt_p1[:rank,]

        u_p2, s_p2, vt_p2 = np.linalg.svd(no_p2_raw, full_matrices=False)
        phi_p2 = vt_p2[:rank,]

        u_all, s_all, vt_all = np.linalg.svd(no_all_raw, full_matrices=False) # (480, 32) (32,) (32, 32)
        phi_all = vt_all[:rank,] # (rank, 32)   

        no_p1_reshape = no_p1.reshape(w * h * s, t) # (30720, 32)
        space_coefficient_p1 = np.matmul(no_p1_reshape, phi_p1.T.conj()) # (30720, 8)

        no_p2_reshape = no_p2.reshape(w * h * s, t)
        space_coefficient_p2 = np.matmul(no_p2_reshape, phi_p2.T.conj())

        # Split complex values into real and imaginary channels.
        space_coefficient_p1_real = np.real(space_coefficient_p1).astype(np.float32)
        space_coefficient_p1_imag = np.imag(space_coefficient_p1).astype(np.float32) # (30720, 8)
        space_coefficient_p1_comp = np.stack((space_coefficient_p1_real, space_coefficient_p1_imag), axis=-1) # (30720, 8, 2)
        space_coefficient_p1 = space_coefficient_p1_real + 1j * space_coefficient_p1_imag # (30720, 8)

        space_coefficient_p2_real = np.real(space_coefficient_p2).astype(np.float32)
        space_coefficient_p2_imag = np.imag(space_coefficient_p2).astype(np.float32)
        space_coefficient_p2 = space_coefficient_p2_real + 1j * space_coefficient_p2_imag

        phi_p1_real = np.real(phi_p1).astype(np.float32)
        phi_p1_imag = np.imag(phi_p1).astype(np.float32) # (8, 32)
        phi_p1 = phi_p1_real + 1j * phi_p1_imag # (8, 32)

        phi_p2_real = np.real(phi_p2).astype(np.float32)
        phi_p2_imag = np.imag(phi_p2).astype(np.float32)
        phi_p2 = phi_p2_real + 1j * phi_p2_imag

        phi_all_real = np.real(phi_all).astype(np.float32)
        phi_all_imag = np.imag(phi_all).astype(np.float32)
        phi_all = phi_all_real + 1j * phi_all_imag

        # Reshape tensors.
        space_coefficient_p1_resha = space_coefficient_p1_comp.reshape(w * h, s, rank, -1) # (256, 120, 8, 2)

        # Build the network input tensor.
        space_coefficient_p1_input = torch.from_numpy(np.expand_dims(space_coefficient_p1_resha, 0)) # 1, 256, 120, 8, 2
        space_coefficient_p1_input = space_coefficient_p1_input.permute([0, 4, 3, 1, 2]) # (1, 2, 8, 256, 120)
        space_coefficient_p1_input = space_coefficient_p1_input.to(device)

        # Move tensors to the selected device.
        space_coefficient_p1 = torch.from_numpy(space_coefficient_p1)
        space_coefficient_p2 = torch.from_numpy(space_coefficient_p2) # (30720, 8)

        space_coefficient_p1 = space_coefficient_p1.to(device)
        space_coefficient_p2 = space_coefficient_p2.to(device)

        phi_p1 = torch.from_numpy(phi_p1).to(device) # (8, 32)
        phi_p2 = torch.from_numpy(phi_p2).to(device)
        phi_all = torch.from_numpy(phi_all).to(device)

        t0 = time.time()

        # Forward pass.
        space_coefficient_p1_pre = denoise_generator(space_coefficient_p1_input) # (1, 2, 8, 256, 120)

        t1 = time.time()
        train_time = t1 - t0
        epoch_time += train_time

        # Remove singleton dimensions.
        space_coefficient_p1_pre = space_coefficient_p1_pre.squeeze() # (2, 8, 256, 120)

        # Reorder dimensions for complex reconstruction.
        space_coefficient_p1_pre = space_coefficient_p1_pre.permute([0, 2, 3, 1]) # (2, 256, 120, 8)

        # Recombine real and imaginary channels.
        space_coefficient_p1_pre = space_coefficient_p1_pre[0, :, :, :] + 1j * space_coefficient_p1_pre[1, :, :, :] # (256, 120, 8)
        space_coefficient_p1_pre = space_coefficient_p1_pre.reshape(w, h, s, -1) # (16, 16, 120, 8)
        space_coefficient_p1_pre = space_coefficient_p1_pre.reshape(w * h * s, -1) # (30720, 8)

        # Reconstruct low-rank images.
        no_p1_lr_pre = torch.matmul(space_coefficient_p1_pre, phi_all) # (30720, 8) (8, 32) > (30720, 32)
        no_p1_lr = torch.matmul(space_coefficient_p1, phi_p1)
        no_p2_lr = torch.matmul(space_coefficient_p2, phi_p2)

        # Reshape tensors.
        no_p1_lr_pre = no_p1_lr_pre.reshape(w, h, s, t) # (16, 16, 120, 32)
        no_p1_lr = no_p1_lr.reshape(w, h, s, t)
        no_p2_lr = no_p2_lr.reshape(w, h, s, t)

        no_p1_lr_pre_real = no_p1_lr_pre.real
        no_p1_lr_pre_imag = no_p1_lr_pre.imag
        no_p1_lr_pre_re_im = torch.stack((no_p1_lr_pre_real, no_p1_lr_pre_imag), dim=0) # (2, 16, 16, 120, 32)

        no_p2_lr_real = no_p2_lr.real
        no_p2_lr_imag = no_p2_lr.imag
        no_p2_lr_re_im = torch.stack((no_p2_lr_real, no_p2_lr_imag), dim=0) # (2, 16, 16, 120, 32)

        # Pixel-wise loss
        L2_loss = L2_pixelwise(no_p1_lr_pre_re_im, no_p2_lr_re_im)

        optimizer_G.zero_grad()

        # Total loss
        Total_loss = L2_loss

        Total_loss.backward()
        optimizer_G.step()

        if index == 0 or index == 1:

            no_p1_lr_pre_re_im_iz12[index, ...] = no_p1_lr_pre_re_im.detach().cpu().numpy()

            no_p1_lr_pre_re_im_iz12_mean = np.mean(no_p1_lr_pre_re_im_iz12, axis=0)
            no_p1_lr_pre_re_im_iz12_mean_comp = no_p1_lr_pre_re_im_iz12_mean[0, ...] + 1j * \
                                                no_p1_lr_pre_re_im_iz12_mean[1, ...]
        
        if index == 0 or index == 1 or index == 3:
            
            print("\n")
            sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %f, L2 Loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                index,
                len(coordinate_list),
                Total_loss.item(),
                L2_loss.item()
            )
            )

            writer.add_scalar('Loss/Total_loss', Total_loss.item(), n)
            writer.add_scalar('Loss/L2_loss', L2_loss.item(), n)

            fig_pre = plot_cpu(no_p1_lr_pre.detach().cpu().numpy())
            writer.add_figure(f"1_de", fig_pre, n)
            plt.close(fig_pre)  

            fig_de_iz12 = plot_cpu(no_p1_lr_pre_re_im_iz12_mean_comp)
            writer.add_figure(f'2_de_iz12', fig_de_iz12, n)
            plt.close(fig_de_iz12)  

            fig_gt = plot_cpu(gt)
            writer.add_figure(f'3_gt', fig_gt, n)
            plt.close(fig_gt)  

            fig_no_all = plot_cpu(no_all)
            writer.add_figure(f'4_no_all', fig_no_all, n)
            plt.close(fig_no_all)    

            fig_np2_lr = plot_cpu(no_p2_lr.detach().cpu().numpy())
            writer.add_figure(f'7_np2_lr', fig_np2_lr, n)
            plt.close(fig_np2_lr)     

            fig_np1_lr = plot_cpu(no_p1_lr.detach().cpu().numpy())
            writer.add_figure(f'5_np1_lr', fig_np1_lr, n)
            plt.close(fig_np1_lr)                

            fig_np1 = plot_cpu(no_p1)
            writer.add_figure(f'6_np1', fig_np1, n)
            plt.close(fig_np1)            

            fig_np2 = plot_cpu(no_p2)
            writer.add_figure(f'8_np2', fig_np2, n)
            plt.close(fig_np2)

            n += 1

        if index == 1:
            denoise_img = no_p1_lr_pre_re_im_iz12_mean_comp
            de = os.path.join(train_result_path, logname + '_epoch_' + str(epoch) + '.mat')
            scio.savemat(de, {'de': denoise_img})

            gt_abs_norm = abs(gt) / abs(gt).max()
            no_p1_lr_pre_re_im_iz12_mean_comp_abs_norm = abs(no_p1_lr_pre_re_im_iz12_mean_comp) / abs(no_p1_lr_pre_re_im_iz12_mean_comp).max()
            iz1_abs_norm = abs(no_p1_lr_pre_re_im_iz12[0,...]) / abs(no_p1_lr_pre_re_im_iz12[0,...]).max()
            iz2_abs_norm = abs(no_p1_lr_pre_re_im_iz12[1,...]) / abs(no_p1_lr_pre_re_im_iz12[1,...]).max()

            rmse_iz12 = np.sqrt(np.mean((no_p1_lr_pre_re_im_iz12_mean_comp_abs_norm[:,:,49:68,:] - gt_abs_norm[:,:,49:68,:])**2))
            rmse_iz1 = np.sqrt(np.mean((iz1_abs_norm[0,:,:,49:68,:] - gt_abs_norm[:,:,49:68,:])**2))
            rmse_iz2 = np.sqrt(np.mean((iz2_abs_norm[1,:,:,49:68,:] - gt_abs_norm[:,:,49:68,:])**2))

    train_time_sum += epoch_time

    # Append epoch metrics to the training log.
    with open(os.path.join(checkpoint_path, 'output.txt'), 'a') as f:
        print(f"epoch: {epoch} time: {epoch_time:.6f}  \
            \n rmse_iz12: {rmse_iz12:.6f} rmse_iz1: {rmse_iz1:.6f} rmse_iz2: {rmse_iz2:.6f}")
        
        f.write(f"epoch: {epoch} time: {epoch_time:.6f} \
                \n rmse_iz12: {rmse_iz12:.6f} rmse_iz1: {rmse_iz1:.6f} rmse_iz2: {rmse_iz2:.6f}\n")

    if epoch % 1 == 0:
        torch.save(denoise_generator.state_dict(), os.path.join(checkpoint_path, 'epoch_' + str(epoch) + '.pth'))

torch.save(denoise_generator.state_dict(), os.path.join(checkpoint_path, 'epoch_' + str(opt.n_epochs) + '.pth'))

time_end = time.time()
all_time = time_end - time_start

writer.close()

print("all_time_sum: {}s, all_time_aver: {}s".format(all_time , all_time / 20))
print("train_time_sum: {}s, train_time_aver: {}s".format(train_time_sum , train_time_sum / 20))