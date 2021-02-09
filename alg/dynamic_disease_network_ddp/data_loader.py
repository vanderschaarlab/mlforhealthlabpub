"""
The data loader is partially adapted from https://github.com/HMEIatJHU/neurawkes
"""

import numpy as np
import torch
import random
import pickle


def get_fold(dat_dict, fold=5, seed=666):
    dat_list = dat_dict['train']
    dim_process = dat_dict['dim_process']
    max_len = dat_dict['max_len']
    random.seed(seed)
    total_samples = len(dat_list)
    eids = list(range(total_samples))
    eid_set = set(eids)
    random.shuffle(eids)

    fold_list = list()
    for i in range(fold):
        eid_test = eids[i::fold]
        dat_test_fold = [dat_list[k] for k in eid_test]

        eid_remain = list(eid_set - set(eid_test))
        eid_val = eid_remain[::10]
        dat_val_fold = [dat_list[k] for k in eid_val]
        eid_train = list(set(eid_remain) - set(eid_val))

        dat_train_fold = [dat_list[k] for k in eid_train]
        fold_dict = {'train': dat_train_fold, 'dev': dat_val_fold, 'test': dat_test_fold,
                     'dim_process': dim_process, 'max_len': max_len}
        fold_list.append(fold_dict)

    return fold_list


def process_seq(data, list_idx_data, max_len, n_event_type, tag_batch='train', dtype=np.float32):
    size_batch = len(list_idx_data)

    # initialize everything with zeros
    seq_time_to_end_np = np.zeros(
        (max_len, size_batch), dtype=dtype
    )
    seq_time_to_current_np = np.zeros(
        (max_len, max_len, size_batch), dtype=dtype
    )
    seq_type_event_np = np.zeros(
        (max_len, size_batch), dtype=np.int64
    )
    time_since_start_to_end_np = np.zeros(
        (size_batch,), dtype=dtype
    )
    seq_mask_np = np.zeros(
        (max_len, size_batch), dtype=dtype
    )
    seq_mask_to_current_np = np.zeros(
        (max_len, max_len, size_batch), dtype=dtype
    )
    num_events_start_to_end_np = np.zeros(
        (size_batch,), dtype=dtype
    )
    event_time_to_end_np = np.zeros((n_event_type, size_batch), dtype=dtype)
    intensity_mask_np = np.ones((n_event_type, max_len, size_batch), dtype=dtype)
    # loop and config data
    for idx_in_batch, idx_data in enumerate(list_idx_data):
        seq = data[tag_batch][idx_data]
        if len(seq) > 0:
            time_end = seq[-1]['time_since_start']
            time_since_start_to_end_np[
                idx_in_batch
            ] = time_end
            num_events_start_to_end_np[
                idx_in_batch
            ] = np.float32(len(seq))
        else:
            time_end = np.float32(0.0)
            time_since_start_to_end_np[
                idx_in_batch
            ] = time_end
            num_events_start_to_end_np[
                idx_in_batch
            ] = np.float32(1)
        for idx_pos, item_event in enumerate(seq):
            t_i = item_event['time_since_start']
            seq_time_to_end_np[
                idx_pos, idx_in_batch
            ] = time_end - t_i
            event_time_to_end_np[
                item_event['type_event'], idx_in_batch
            ] = time_end - t_i
            seq_type_event_np[
                idx_pos, idx_in_batch
            ] = item_event['type_event']
            seq_mask_np[
                idx_pos, idx_in_batch
            ] = np.float32(1.0)
            intensity_mask_np[
            item_event['type_event'], idx_pos:, idx_in_batch
            ] = np.float32(0.0)

            idx_pos_prime = 0
            while idx_pos_prime < idx_pos:
                item_event_prime = seq[idx_pos_prime]
                t_i_prime = item_event_prime[
                    'time_since_start'
                ]
                seq_time_to_current_np[
                    idx_pos, idx_pos_prime, idx_in_batch
                ] = t_i - t_i_prime
                seq_mask_to_current_np[
                    idx_pos, idx_pos_prime, idx_in_batch
                ] = np.float32(1.0)
                idx_pos_prime += 1

    return seq_time_to_end_np, seq_time_to_current_np, seq_type_event_np, time_since_start_to_end_np, seq_mask_np, \
           seq_mask_to_current_np, intensity_mask_np, event_time_to_end_np


def get_train_test_split(proportion_train, batch_input):
    n_record = batch_input[0].shape[1]
    train_idx = np.random.uniform(0, 1, n_record) <= proportion_train
    train_batch = map(lambda x: x[..., train_idx], batch_input)
    test_batch = map(lambda x: x[..., np.logical_not(train_idx)], batch_input)

    return list(train_batch), list(test_batch)


def get_mini_batch(batch_size, batch_input):
    n_record = batch_input[0].shape[1]
    idx = np.random.choice(list(range(n_record)), size=batch_size, replace=False)
    mini_batch = map(lambda x: torch.tensor(x[..., idx]), batch_input)
    return list(mini_batch)


def get_whole_batch(batch_input):
    n_record = batch_input[0].shape[1]
    idx = list(range(n_record))
    mini_batch = map(lambda x: torch.tensor(x[..., idx]), batch_input)
    return list(mini_batch)


def get_partition_batch(n_partition, batch_input):
    n_record = batch_input[0].shape[1]
    idx = list(range(n_record))
    random.shuffle(idx)
    idx_list = [idx[i::n_partition] for i in range(n_partition)]
    res_list = []
    for id in idx_list:
        batch = list(map(lambda x: torch.tensor(x[..., id]), batch_input))
        res_list.append(batch)
    return res_list


def get_partition_fixed_batch(batch_size, batch_input):
    n_record = batch_input[0].shape[1]
    n_partition = n_record // batch_size
    idx = list(range(n_record))
    random.shuffle(idx)
    idx_list = [idx[i * batch_size:(i + 1) * batch_size] for i in range(n_partition)]
    res_list = []
    for id in idx_list:
        batch = list(map(lambda x: torch.tensor(x[..., id]), batch_input))
        res_list.append(batch)
    return res_list


def get_data(data_path):
    tag_list = ['train', 'dev', 'test']
    data_dict = {}

    with open(data_path, 'rb') as f:
        data_temp = pickle.load(f)

    dim_process = data_temp['dim_process']
    max_len = data_temp['max_len']

    for tag in tag_list:
        first_occurrence_only = False

        data = process_seq(data_temp, range(len(data_temp[tag])), max_len, dim_process, tag_batch=tag,
                           dtype=np.float32)
        context_mat = np.zeros((1, data[1].shape[2]), dtype=np.float32)
        data = list(data)
        data.append(context_mat)
        data_dict[tag] = data
    return data_dict, max_len, dim_process
