import numpy as np
import os
import random
import h5py


def _find_mat_file(data_path, include_keywords, exclude_keywords=()):
    matched_files = []
    for root, _, files in os.walk(data_path):
        for file_name in files:
            lower_name = file_name.lower()
            if not lower_name.endswith('.mat'):
                continue
            if all(keyword in lower_name for keyword in include_keywords) and not any(keyword in lower_name for keyword in exclude_keywords):
                matched_files.append(os.path.join(root, file_name))

    matched_files.sort()
    if len(matched_files) != 1:
        raise FileNotFoundError(
            f"Expected one .mat file in {data_path} matching include={include_keywords}, "
            f"exclude={exclude_keywords}, found {matched_files}"
        )
    return matched_files[0]


def get_train_no_ksp_path(data_path):
    return _find_mat_file(data_path, include_keywords=('no_ksp',), exclude_keywords=('all',))


def get_train_no_ksp_logname(data_path):
    no_ksp_path = get_train_no_ksp_path(data_path)
    return os.path.splitext(os.path.basename(no_ksp_path))[0]


def train_preprocess_lessMemoryMulStacks(args):

    coordinate_list={}

    no_ksp_path = get_train_no_ksp_path(args.data_path)
    no_ksp = h5py.File(no_ksp_path)
    no_ksp = np.transpose(no_ksp['no_n2n_ksp'])
    no_ksp = no_ksp['real'] + 1j * no_ksp['imag']
    no_ksp = no_ksp / np.abs(no_ksp.max())
    no_ksp = no_ksp.transpose(0,1,3,4,2).astype(np.complex64) # (64, t, 32, 32, 120)

    no_ksp_all_path = _find_mat_file(args.data_path, include_keywords=('no_ksp', 'all'))
    no_ksp_all = h5py.File(no_ksp_all_path)
    no_ksp_all = no_ksp_all['no_ksp'] 
    no_ksp_all = no_ksp_all['real'] + 1j * no_ksp_all['imag'] # (32, 32, 120, 32)
    no_ksp_all = no_ksp_all / np.abs(no_ksp_all.max())
    no_ksp_all = no_ksp_all.transpose(1,0,2,3).astype(np.complex64)

    gt_ksp_path = _find_mat_file(args.data_path, include_keywords=('gt_ksp',))
    gt_ksp = h5py.File(gt_ksp_path)
    gt_ksp = gt_ksp['gt_ksp'] 
    gt_ksp = gt_ksp['real'] + 1j * gt_ksp['imag']
    gt_ksp = gt_ksp / np.abs(gt_ksp.max())
    gt_ksp = gt_ksp.transpose(1,0,2,3).astype(np.complex64) # (32,32,120,32)

    num_list = []
    for i in range(0, 64):
        for j in range(0, 64):
            if i != j:
                num_pair = [i, j]
                num_list.append(num_pair)
    coordinate_list = num_list

    return  coordinate_list, no_ksp, no_ksp_all, gt_ksp

def shuffle_datasets_lessMemory(name_list):
    index_list = list(range(0, len(name_list)))
    random.shuffle(index_list)
    random_index_list = index_list
    new_name_list = list(range(0, len(name_list)))
    for i in range(0,len(random_index_list)):
        new_name_list[i] = name_list[random_index_list[i]]
    return new_name_list


def test_preprocess_lessMemoryNoTail (args):

    no_ksp = h5py.File(get_train_no_ksp_path(args.data_path))
    no_ksp = np.transpose(no_ksp['no_n2n_ksp'])
    no_ksp = no_ksp['real'] + 1j * no_ksp['imag']
    no_ksp = no_ksp / np.abs(no_ksp.max())
    no_ksp = no_ksp.transpose(0,1,3,4,2).astype(np.complex64)[0:2,...] # (64, t, 32, 32, 120)

    no_ksp_all = h5py.File(_find_mat_file(args.data_path, include_keywords=('no_ksp', 'all')))
    no_ksp_all = no_ksp_all['no_ksp'] 
    no_ksp_all = no_ksp_all['real'] + 1j * no_ksp_all['imag'] # (32, 32, 120, 32)
    no_ksp_all = no_ksp_all / np.abs(no_ksp_all.max())
    no_ksp_all = no_ksp_all.transpose(1,0,2,3).astype(np.complex64)

    return no_ksp, no_ksp_all