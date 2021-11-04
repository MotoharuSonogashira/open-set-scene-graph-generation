import os, glob, pickle, json
import numpy as np
import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.data import datasets as D


def gather(a, indices, axis=-1):
    if axis < 0:
        axis = a.ndim + axis
    return a[tuple(indices if i == axis else np.arange(a.shape[i])
        for i in range(a.ndim))]

def load(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.pytorch':
        return torch.load(filename, map_location='cpu')
    elif ext == '.json':
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)

def save(filename, obj):
    _, ext = os.path.splitext(filename)
    if ext == '.pytorch':
        torch.save(obj, filename)
    elif ext == '.json':
        with open(filename, 'w') as f:
            json.dump(obj, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

def analyze_method_dir(method_dir, allow_missing=False):
    if not os.path.exists(method_dir):
        if allow_missing:
            return None, None, None
        raise RuntimeError(f"directory not found: '{method_dir}'")
    method_name = os.path.basename(method_dir)

    inference_dir = os.path.join(method_dir, 'inference')
    if not os.path.exists(inference_dir):
        if allow_missing:
            return method_name, None, None
        raise RuntimeError(f"no inference directory in '{method_dir}'")

    result_dirs = glob.glob(os.path.join(inference_dir, '*'))
    if len(result_dirs) != 1:
        if allow_missing:
            return method_name, None, None
        raise RuntimeError("no" if len(result_dirs) == 0 else "multiple" +
                f" result directories in '{inference_dir}'")
    result_dir = result_dirs[0]
    dataset_name = os.path.basename(result_dir)
    return method_name, dataset_name, result_dir


def load_dataset(dataset_name):
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
    data = paths_catalog.DatasetCatalog.get(dataset_name, cfg)
    dataset = getattr(D, data["factory"])(**data["args"])

    for k in '__unknown__', '__missing__':
        if k not in dataset.classes_to_ind:
            # add the unknown/missing class if it is not present in the dataset
            dataset.ind_to_classes.append(k)
            dataset.classes_to_ind[k] = dataset.ind_to_classes.index(k)
    return dataset

def get_groundtruths(dataset):
    return [dataset.get_groundtruth(image_id, evaluation=True)
            for image_id in range(len(dataset))]


def load_eval_results(path):
    filename = os.path.join(path, 'eval_results_shrunk.pytorch')
    if not os.path.exists(filename):
        filename = os.path.join(path, 'eval_results.pytorch')
    eval_results = load(filename)

    if eval_results['predictions'][0].get_field('pred_rel_scores').ndim == 2:
        # keep the score of selected class only
        for boxlist in eval_results['predictions']:
            fields = boxlist.extra_fields
            boxlist.extra_fields['pred_rel_scores'] = gather(
                fields['pred_rel_scores'], fields['pred_rel_labels'], axis=1)
    return eval_results
