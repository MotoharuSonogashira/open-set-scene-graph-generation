#!/usr/bin/env python3

import argparse, os, pickle
import numpy as np
import pandas as pd
from more_itertools import first
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import bbox_overlaps, intersect_2d
from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import _triplet
from utils import load, save, \
        analyze_method_dir, load_dataset, load_eval_results


def match_boxes(boxes_1, boxes_2, iou_thres):
    assert boxes_1.ndim == boxes_2.ndim == 2
    assert boxes_1.shape[1] == boxes_2.shape[1]
    assert boxes_1.shape[1] % 4 == 0
    n = boxes_1.shape[1] // 4
        # number of objects (1/2 for object/relationships counts)
    boxes_1 = np.split(boxes_1, n, 1)
    boxes_2 = np.split(boxes_2, n, 1)
    ious = [bbox_overlaps(bs_1, bs_2) for bs_1, bs_2 in zip(boxes_1, boxes_2)]
    return (np.stack(ious, axis=-1) >= iou_thres).all(-1) # over objects

def compute_statistics(gt_classes, pred_classes, gt_boxes, pred_boxes,
        iou_thres, unknown_class, object_class_indices):
        # count gts for each number of unknowns and for each classification
        # against preds;
        # assume that the predictions (pred_classes and pred_boxes) are sorted
        # by scores
    assert gt_classes.ndim == pred_classes.ndim
    class_match = intersect_2d(gt_classes, pred_classes)     # (#gts, #preds)
    box_match = match_boxes(gt_boxes, pred_boxes, iou_thres) # (#gts, #preds)
    assert class_match.shape == box_match.shape
    num_gt_unknowns = (
            gt_classes[:, object_class_indices] == unknown_class).sum(1)

    result_dict = {20: {}, 50: {}, 100: {}}
    for k in result_dict.keys():
        mask = box_match[:, :k] # consider top-k preds only

        results = {n: {'correct': 0, 'wrong': 0, 'background': 0}
            for n in range(len(object_class_indices) + 1)}
            # for each number of unknown objects (class labels corresponding to
            # objects along the 2nd axis of gt_classes are indicated by
            # object_class_indices)
        for gt_ind, (pred_mask, n) in enumerate(zip(mask, num_gt_unknowns)):
            pred_inds = pred_mask.nonzero()[0]
            c =      'background' if len(pred_inds) == 0                  \
                else 'correct'    if class_match[gt_ind, pred_inds].any() \
                else 'wrong'
            results[n.item()][c] += 1
        result_dict[k] = pd.DataFrame(results)
    return result_dict

def concat(dfs):
    return pd.concat(dfs, keys=range(len(dfs))).unstack().reset_index(drop=True)

def aggregate_df_dicts(df_ds):
    assert all(df.keys() == df_ds[0].keys() for df in df_ds)
    return {k: concat(v) for k, v in
            zip(df_ds[0].keys(), zip(*(dfs.values() for dfs in df_ds)))}


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file',
            default='configs/e2e_relation_X_101_32_8_FPN_1x.yaml')
    parser.add_argument('-C', '--no-cache', dest='cache', action='store_false')
    parser.add_argument('-p', '--precision', type=int, default=2)
    parser.add_argument('-N', '--num', type=int)
    parser.add_argument('path')
    args = parser.parse_args()
    _, dataset_name, result_dir = analyze_method_dir(args.path)

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    # Load dataset info.
    dataset = load_dataset(dataset_name)
    num_obj_classes = len(dataset.ind_to_classes   )
    num_rel_classes = len(dataset.ind_to_predicates)
    unknown_class = dataset.classes_to_ind['__unknown__']
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD

    # Load data.
    eval_results = load_eval_results(result_dir)
    groundtruths = eval_results['groundtruths']
    predictions  = eval_results['predictions' ]

    # Compute statistics for each image.
    obj_filename = os.path.join(result_dir, 'obj_stats.pkl')
    rel_filename = os.path.join(result_dir, 'rel_stats.pkl')
    if args.cache and os.path.exists(obj_filename):
        obj_stats = load(obj_filename)
        rel_stats = load(rel_filename)
    else:
        obj_stats = []
        rel_stats = []
        image_ids = list(range(len(groundtruths)))
        if args.num is not None:
            image_ids = image_ids[:args.num]
        for image_id in tqdm(image_ids):
            groundtruth = groundtruths[image_id]
            prediction  = predictions [image_id]
            gt_boxes         = groundtruth.bbox                       .numpy()
            gt_classes       = groundtruth.get_field('labels'        ).numpy()
            gt_rels          = groundtruth.get_field('relation_tuple').numpy()
            pred_boxes       = prediction.bbox                        .numpy()
            pred_classes     = prediction.get_field('pred_labels'    ).numpy()
            obj_scores       = prediction.get_field('pred_scores'    ).numpy()
            pred_rel_inds    = prediction.get_field('rel_pair_idxs'  ).numpy()
            pred_rel_classes = prediction.get_field('pred_rel_labels').numpy()

            # Compute object statistics.
            assert gt_classes.ndim == pred_classes.ndim == 1
            inds = np.argsort(obj_scores, axis=0)[::-1] # descending
            sorted_pred_boxes   = pred_boxes  [inds]
            sorted_pred_classes = pred_classes[inds]
                # sort predicted objects here (not to be used for relationships
                # because they assume the original order)
            obj_stats.append(compute_statistics(
                    gt_classes         [:, np.newaxis],
                    sorted_pred_classes[:, np.newaxis],
                    gt_boxes, sorted_pred_boxes, iou_thres, unknown_class, [0]))
                # class arrays must be 2D (as in relationship counts)

            # Compute relationship statistics.
            pred_rels = np.column_stack((pred_rel_inds, pred_rel_classes))
            gt_triplets, gt_triplet_boxes, _ = _triplet(
                    gt_rels, gt_classes, gt_boxes)
            pred_triplets, pred_triplet_boxes, _ = _triplet(
                    pred_rels, pred_classes, pred_boxes)
                # predicted relationships are already sorted in
                # relation_head.PostProcessor
            rel_stats.append(compute_statistics(gt_triplets, pred_triplets,
                    gt_triplet_boxes, pred_triplet_boxes,
                    iou_thres, unknown_class, [0, 2]))
                # consider subject and object classes only in unknown counting
                # for each GT

        obj_stats = aggregate_df_dicts(obj_stats)
        rel_stats = aggregate_df_dicts(rel_stats)
        save(obj_filename, obj_stats)
        save(rel_filename, rel_stats)

    pd.set_option('display.float_format', f'{{:.{args.precision}f}}'.format)
    dfs = {}
    for k in obj_stats.keys():
        df = pd.concat({name: df.mean(axis=0).unstack().T
            for name, df in [('obj', obj_stats[k]), ('rel', rel_stats[k])]}
        ).stack().unstack([0, 2]) # two DataFrames side by side
        df = df.rename_axis([f'@{k}', '#unknowns'], axis=1)
        print(df)
        dfs[k] = df

    output = os.path.join(args.path, 'eval')
    os.makedirs(output, exist_ok=True)
    for k, df in dfs.items():
        df.to_pickle(os.path.join(output, f'{k}.pkl'))

if __name__ == '__main__':
    main()
