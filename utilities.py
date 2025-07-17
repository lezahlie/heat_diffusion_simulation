from setup_logger import setup_logger, logging, global_logger as logger
from json import loads, dump, load, dumps
from multiprocessing import Process, Manager, Lock, current_process
from os import makedirs, path as os_path, remove as os_remove, getpid, cpu_count, remove, listdir, environ, rename

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimate
import matplotlib.colors as mcolor
import matplotlib.patches as mpatch
import matplotlib.ticker as mtick

import numpy as np
import pandas as pd
from scipy import ndimage, sparse

import re
import h5py
import warnings
from glob import glob
from pprint import pprint
from typing import Dict, List, Tuple, Mapping, Any, Optional


DEFAULT_DATAFILE_EXT = "hdf5"
DATATYPE_NAME = "heat_diffusion"


def create_save_states_predicate(conditions: List[str]|None):
    combined_predicate = lambda i: False

    if not conditions:
        return combined_predicate

    if 'all' in conditions:
        return lambda i: True

    for condition in conditions:
        
        # Handle 'first-N'
        if isinstance(condition, tuple) and condition[0] == 'first':
            n = condition[1]
            # Combine with OR: (1 <= i <= n) or the existing combined_predicate
            combined_predicate = lambda i, combined_predicate=combined_predicate, n=n: (1 <= i <= n) or combined_predicate(i)

        # Handle 'interval-T'
        elif isinstance(condition, tuple) and condition[0] == 'interval':
            t = condition[1]
            # Combine with OR: (i % t == 0) or the existing combined_predicate
            combined_predicate = lambda i, combined_predicate=combined_predicate, t=t: (i % t == 0) or combined_predicate(i)

        # Handle 'base-N'
        elif isinstance(condition, tuple) and condition[0] == 'base':
            b = condition[1]
            def is_power(i, num):
                if i < 1:
                    return False
                while i % num == 0:
                    i //= num
                return i == 1
            # Combine with OR: is_power(i, b) or the existing combined_predicate
            combined_predicate = lambda i, combined_predicate=combined_predicate, b=b, is_power=is_power: is_power(i, b) or combined_predicate(i)

    return combined_predicate


# creates a folder path if it doesn't exist
def create_folder(folder_name):
    if os_path.isdir(folder_name):
        return os_path.abspath(folder_name)
    folder_path=os_path.dirname(os_path.abspath(__file__))
    dir_path = os_path.join(folder_path, folder_name)
    makedirs(dir_path, exist_ok=True)
    if not os_path.exists(dir_path):
        logger.error(f"Cannot create folder '{folder_name}' in path: {folder_path}", stacklevel=2)
    return dir_path


# returns a path to a file, creates the the root directory if needed
def create_file_path(folder_path, file_name):
    out_path = create_folder(folder_path)
    file_path = os_path.join(out_path, file_name)
    return file_path


# removes a file if it exists
def remove_if_exists(file_path=None):
    if file_path and os_path.exists(file_path):
        remove(file_path) 


# converts dataframes to lists of dictionary records and vice versa
def dataframe_dictrecord_converter(data):
    if isinstance(data, pd.DataFrame):
        dict_record = data.to_dict(orient='records')
        return dict_record
    elif isinstance(data, list) and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        return df
    else:
        raise TypeError(f"Expected types 'pd.Dataframe' or 'dict[]' for conversion, but recieved type '{type(data)}'")


# splits ranges of seeds based on step size, meant for data splitting between tasks
def split_seed_range(seed_range, seed_step):
    min_seed, max_seed = seed_range
    return [(start_seed, min(start_seed + seed_step - 1, max_seed)) for start_seed in range(min_seed, max_seed + 1, seed_step)]


# flattens nested dictionaries, by appending top level key as a prefix
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# shared dictionary between processes
def update_shared_data(new_data, shared_data, shared_lock):
    with shared_lock:
        shared_data_copy = shared_data.copy()

    for category in ['image', 'meta', 'metric', 'scalar']:
        if category not in new_data:
            continue
        if category not in shared_data_copy:
            shared_data_copy[category] = {}

        for key, new_stats in new_data[category].items():
            if key not in shared_data_copy[category]:
                shared_data_copy[category][key] = new_stats
            else:
                original_stats = shared_data_copy[category][key]

                # Calculate new min and max
                new_min = min(original_stats['min'], new_stats["min"])
                new_max = max(original_stats['max'], new_stats["max"])

                # Calculate the new count (total number of elements)
                new_count = original_stats['count'] + new_stats["count"]

                # Calculate the pooled mean
                new_mean = (original_stats["count"] * original_stats["mean"] + new_stats["count"] * new_stats["mean"]) / new_count

                # Calculate the pooled variance
                pooled_variance = (
                    (original_stats["count"] - 1) * original_stats["std"] ** 2 +
                    (new_stats["count"] - 1) * new_stats["std"] ** 2 +
                    (original_stats["count"] * new_stats["count"]) / (new_count) * (original_stats["mean"] - new_stats["mean"]) ** 2
                ) / (new_count - 1)

                # Calculate the pooled standard deviation
                new_std = np.sqrt(pooled_variance)

                new_shape = [original_stats['shape'][0] + new_stats['shape'][0]] + original_stats['shape'][1:]

                # Update the shared data dictionary with pooled values
                shared_data_copy[category][key] = {
                    'min': new_min,
                    'max': new_max,
                    'mean': new_mean,
                    'std': new_std,
                    'count': new_count,
                    'shape': new_shape
                }

    with shared_lock:
        shared_data.update(shared_data_copy)


def compute_local_stats(sim_results_list):
    stats_dict = {}
    data_groups = ['image', 'metric']  # Track image and metric data

    for sim_results in sim_results_list:
        for group in data_groups:
            if group not in sim_results:
                continue

            if group not in stats_dict:
                stats_dict[group] = {}

            for key, value in sim_results[group].items():
                data = np.array(value)
                local_min = np.min(data)
                local_max = np.max(data)
                local_mean = np.mean(data)
                local_std = np.std(data)
                local_size = data.size 
                local_shape = list(data.shape) if group == "image" else [1]

                if key not in stats_dict[group]:
                    stats_dict[group][key] = {
                        'min': local_min,
                        'max': local_max,
                        'mean': local_mean,
                        'std': local_std,
                        'count': local_size,
                        'shape': [1] + local_shape
                    }
                else:
                    stats_dict[group][key]['min'] = min(stats_dict[group][key]['min'], local_min)
                    stats_dict[group][key]['max'] = max(stats_dict[group][key]['max'], local_max)
                    stats_dict[group][key]['mean'] = local_mean
                    stats_dict[group][key]['std'] = local_std
                    stats_dict[group][key]['count'] += local_size
                    stats_dict[group][key]['shape'][0] += 1

    return stats_dict


def extract_minmax_tuples(global_stats):
    min_max_dict = {}
    for category, values in global_stats.items():
        min_max_dict[category] = {}
        for key, stats in values.items():
            min_max_dict[category][key] = (stats["min"], stats["max"])
    return min_max_dict


# serializes numpy types bc ya know
def serialize_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: serialize_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    else:
        return obj


# writes or appends to a csv file
def save_to_json(file_path, content_in:dict, mode='w', indent=4):
    try:
        content_out = serialize_numpy_types(content_in)
        with open(file_path, mode) as json_file:
            dump(content_out, json_file, indent=indent)
    except Exception as e:
        logger.error(e, stacklevel=2)   


# reads json file
def read_from_json(file_path):
    try:
        with open(file_path, 'r') as json_file:
            content = load(json_file)
        return content
    except Exception as e:
        logger.error(e, stacklevel=2)   


# writes or appends to a hdf5 file
def save_to_hdf5(data_dict_list, file_path, chunk_size=None, flatten=False):
    def write_data_to_group(group, data):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                group.attrs[key] = value
                #logger.debug(f"Saved attribute: {key} => {value}")
            elif isinstance(value, str):
                string_dt = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(key, data=value, dtype=string_dt)
                #logger.debug(f"Saved string dataset: {key} => {value}")
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
                #logger.debug(f"Saved array dataset: {key} => array with shape {value.shape}")
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                write_data_to_group(subgroup, value) 
            else:
                raise TypeError(f"Invalid type for {key}: {type(value)}. Expected int, float, str, or np.ndarray.")
            
    chunk = chunk_size or 1
    mode = 'a' if os_path.exists(file_path) else 'w'  
    try:
        with h5py.File(file_path, mode) as f:
            existing_indices = [int(k.split('_')[1]) for k in f.keys() if k.startswith("record_")]
            current_max_index = max(existing_indices) + 1 if existing_indices else 0
            total_records = len(data_dict_list)
            logger.debug(f"Saving {total_records} records to file starting at index {current_max_index}", stacklevel=2)

            for i in range(0, total_records, chunk):
                for idx in range(i, min(i + chunk, total_records)):
                    record_index = current_max_index + idx
                    record_group_name = f"record_{record_index}"

                    if record_group_name in f:
                        logger.debug(f"Skipping existing group: {record_group_name}", stacklevel=2)
                        continue

                    record_dict = flatten_dict(data_dict_list[idx]) if flatten else data_dict_list[idx]
                    record_group = f.create_group(record_group_name)
                    write_data_to_group(record_group, record_dict)
                    logger.debug(f"Created group: {record_group_name}", stacklevel=2)

    except (Exception, OSError, IOError, TypeError) as e:
        logger.error(f"Error writing to HDF5 file {file_path}: {e}", stacklevel=2)


# read a hdf5 file in or a random # of samples 
def read_from_hdf5(file_path, sample_size=None, chunk_size=None, flatten=True, random_seed=None):
    def load_group_data(group):
        group_dict = {}
        group_dict.update({k: v for k, v in group.attrs.items()})

        for key, item in group.items():
            if isinstance(item, h5py.Group):
                subgroup_data = load_group_data(item)
                if flatten:
                    group_dict.update(flatten_dict(subgroup_data, parent_key=key))
                else:
                    group_dict[key] = subgroup_data
            else:
                value = item[()] if item.shape == () else item[:]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
                    value = np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in value])

                group_dict[key] = value

        return group_dict
    
    chunk = chunk_size or 1
    data_dict_list = []
    try:
        with h5py.File(file_path, 'r') as f:
            all_keys = list(f.keys())
            if isinstance(sample_size, int) and (0 < sample_size < len(all_keys)):
                if isinstance(random_seed, int):
                    rng = np.random.default_rng(random_seed)
                else:
                    rng = np.random.default_rng()
                selected_keys = rng.choice(all_keys, sample_size, replace=False)
            else:
                selected_keys = all_keys

            for i in range(0, len(selected_keys), chunk):
                chunk_keys = selected_keys[i:i + chunk]
                for key in chunk_keys:
                    group = f[key]
                    data = load_group_data(group)
                    data_dict_list.append(data)
            return data_dict_list

    except (OSError, IOError, TypeError) as e:
        logger.error(f"Cannot read from HDF5 file: {file_path} due to: {e}", stacklevel=2)


# copies attributes from a HDF5 file
def _copy_attributes(src, dst):
    for attr_key, attr_value in src.attrs.items():
        dst.attrs[attr_key] = attr_value
        logger.debug(f"Copied attribute for {src.name}: {attr_key} => {attr_value}")


# copies datasets from a HDF5 file
def _copy_dataset(src_dataset, dst_group, dataset_name):
    data = src_dataset[()] if src_dataset.shape == () else src_dataset[:]
    if isinstance(data, str):
        dtype = h5py.string_dtype(encoding="utf-8")
        dst_dataset = dst_group.create_dataset(dataset_name, data=data, dtype=dtype)
    else:
        dst_dataset = dst_group.create_dataset(dataset_name, data=data)
    _copy_attributes(src_dataset, dst_dataset)

# copies groups from a HDF5 file
def _copy_group(src_group, dst_group, group_name_prefix=""):
    for key in src_group:
        new_group_name = f"{group_name_prefix}_{key}" if group_name_prefix else key
        item = src_group[key]

        if isinstance(item, h5py.Group):
            dst_subgroup = dst_group.create_group(new_group_name)
            _copy_attributes(item, dst_subgroup)
            _copy_group(item, dst_subgroup)
        elif isinstance(item, h5py.Dataset):
            _copy_dataset(item, dst_group, new_group_name)
        else:
            logger.warning(f"Unexpected item type: {type(item)} for key '{key}'", stacklevel=2)


# combines HDF5 files into a single file, meant for parallel IO
def combine_hdf5_files(input_file_paths, output_file_path, chunk_size=None):
    chunk = chunk_size or 1
    try:   
        with h5py.File(output_file_path, 'a') as dst_file:
            record_index = len(dst_file.keys())
            
            for file_path in input_file_paths:
                with h5py.File(file_path, 'r') as src_file:
                    for record_name in src_file.keys():
                        logger.debug(f"Combining record: {record_name}")

                        unique_record_name = f"record_{record_index}"
                        dst_group = dst_file.create_group(unique_record_name)

                        _copy_attributes(src_file[record_name], dst_group)
                        _copy_group(src_file[record_name], dst_group)

                        record_index += 1
                remove_if_exists(file_path)
                
    except (OSError, IOError, TypeError) as e:
        logger.error(f"Cannot read HDF5 files and combine to one HDF5 file due to: {e}", stacklevel=2)
