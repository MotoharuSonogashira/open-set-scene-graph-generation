#!/usr/bin/env python3

import argparse, os, sys, ast, json, copy
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils import load, analyze_method_dir, load_dataset, load_eval_results

sys.path.append('open-set-scene-graph-dataset/lib') # needed by lib.fast_rcnn.visualize
from fast_rcnn.config import cfg as imp_cfg
    # 'lib' is omitted as it prevents referencing in lib.datasets.viz
from datasets.eval_utils import ground_predictions
from datasets.viz import draw_graph, _viz_scene_graph


def ndargsort(a):
    i = a.flatten().argsort()
        # argsort supports sorting along a single axis only
    grid = np.indices(a.shape).reshape(a.ndim, -1)
    return grid[:, i].T # k-th row is the multiindex of the k-th value in a

def topk_scores(scores, k, value=0):
    scores = scores.copy() # copy to modify
    max_scores = scores.max(-1)
    index_arrays = ndargsort(max_scores) # ascending order
    scores[tuple(index_arrays[:-k].T)] = value # non-top-k objects (pairs)
    return scores

def threshold_scores(scores, threshold, value=0):
    scores = scores.copy() # copy to modify
    scores[(scores.max(axis=-1) < threshold).nonzero()] = value
    return scores

def indices_to_mask(inds, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[inds] = True
    return mask

def make_index_mapping(indices, num):
    indices = np.array(indices)
    assert indices.ndim == 1
    new_indices = np.full((num,), -1)
    new_indices[indices] = np.arange(indices.shape[0])
    return new_indices

def set_unspecified(arr, inds, value):
    mask = indices_to_mask(inds, arr.shape)
    inv_mask = np.logical_not(mask)
    arr[inv_mask.nonzero()] = value


def classes_to_names(classes, hint=None, finalize=True):
    counts = defaultdict(lambda: 0)
    if hint is not None:
        assert len(hint) == len(classes)
        for h in hint:
            if h is None:
                continue
            name, index = h
            counts[name] = max(counts[name], index + 1)
                # initialize the count of each name to its max count + 1 in hint

    # Make pairs of names and counts
    indexed_names = []
    for i, k in enumerate(classes):
        name = imp_cfg.ind_to_class[k]
        if hint is not None and hint[i] is not None and name == hint[i][0]:
            # true count in the hint takes precedence if its name is the same as
            # the predicted name
            count = hint[i][1]
        else:
            # count the prediciton as a new instance of the name otherwise
            count = counts[name]
            counts[name] += 1
        indexed_names.append((name, count))

    if finalize:
        # Suffix the names with the counts.
        multi_instance = defaultdict(lambda: False)
        for name, index in indexed_names:
            if index > 0:
                multi_instance[name] = True
        return [name + (str(index) if multi_instance[name] else '')
            for name, index in indexed_names]
    else:
        return indexed_names

def match_boxes(true_boxes, pred_boxes):
    assert true_boxes.ndim == 2 and pred_boxes.ndim == 2
    assert true_boxes.shape[1] == pred_boxes.shape[1] == 4

    # Map true boxes to pred boxes.
    obj_scores = np.ones((pred_boxes.shape[0], 1))
        # dummy scores for a single class (ground_predictions assumes pred
        # boxes and scores are available for all classes; it only uses
        # the size of the 2nd axis, i.e.,. the number of classes)
    max_overlaps = np.ones((true_boxes.shape[0],))
        # ones for all true boxes (to let ground_predictions select all of them
        # as truth)
    true_to_pred = ground_predictions(
            {'boxes': pred_boxes, 'scores'       : obj_scores  },
            {'boxes': true_boxes , 'max_overlaps': max_overlaps},)
        # all boxes correspond to themselves if true boxes are used as pred

    # Add mappings to None for unmatched true boxes.
    for i in range(len(true_boxes)):
        true_to_pred.setdefault(i, None)
            # avoid KeyError for true boxes without matched preds
    return true_to_pred

def select_objects(boxes, obj_classes, pairs, obj_inds, names=None):
    # Map old indices to new ones.
    new_obj_inds = make_index_mapping(obj_inds, len(obj_classes))

    # Remove objects unspecified by obj_inds.
    boxes       = boxes      [obj_inds]
    obj_classes = obj_classes[obj_inds]
    pairs = new_obj_inds[pairs]
    if names is not None:
        names = [names[i] for i in obj_inds]
        return boxes, obj_classes, pairs, names
    else:
        return boxes, obj_classes, pairs

def select_relationships(pairs, rel_classes, rel_inds):
    pairs       = pairs      [rel_inds]
    rel_classes = rel_classes[rel_inds]
    return pairs, rel_classes

def visualize_scene_graph(img, boxes, obj_classes, pairs, rel_classes,
        image_relationship=False, names=None):
        # img's channels are assumed to be in the RGB order
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert obj_classes.ndim == 1
    assert rel_classes.ndim == 1
    assert boxes.shape[0] == obj_classes.shape[0]
    assert pairs.shape[0] == rel_classes.shape[0]
    assert names is None or len(names) == boxes.shape[0]

    # Extract non-background (i.e., non-filtered-out) objects and
    # relationships.
    # (_viz_scene_graph does this by itself but draw_graph does not)
    obj_inds = (obj_classes != 0).nonzero()[0]
    rel_inds = (rel_classes != 0).nonzero()[0]
    boxes, obj_classes, pairs, names = select_objects(
            boxes, obj_classes, pairs, obj_inds, names=names)
    pairs, rel_classes = select_relationships(pairs, rel_classes, rel_inds)
    assert (pairs >= 0).all()

    # Draw the graph and annotated image.
    rels = np.concatenate([pairs, rel_classes[:, np.newaxis]], axis=1)
    draw_graph(obj_classes, rels, names=names)
    if not image_relationship:
        # hide relationships while showing their objects in viz_scene_graph
        rels[:, 2] = 0
    _viz_scene_graph(img, boxes, obj_classes, rels, preprocess=False,
            names=names)
        # preprocessing (undoing channel-wise normalization and reordering
        # channels into RGB) is unnecessary



def filter_relationships_by_gt(boxes, obj_classes, pairs, rel_classes,
        true_boxes, true_obj_classes, true_pairs, focus_gt_classes=None):
        # modified version of IMP's fast_rcnn.visualize.viz_net and
        # draw_graph_pred
    assert boxes.shape[0] == obj_classes.shape[0]
    assert true_boxes.shape[0] == true_obj_classes.shape[0]

    # Match true and predicted objects.
    true_to_pred = match_boxes(true_boxes, boxes)

    # Filter true relationships.
    true_rel_inds = []
    for ind, (i, j) in enumerate(true_pairs):
        if focus_gt_classes is not None:
            # keep only rels with objs of specified true classes
            if not (   true_obj_classes[i] in focus_gt_classes
                    or true_obj_classes[j] in focus_gt_classes):
                continue
        true_rel_inds.append(ind)
    new_true_pairs = true_pairs[true_rel_inds]
    true_obj_inds = np.unique(new_true_pairs)
        # indices of kept true objects

    # Filter predicted relationships.
    pred_rel_inds = []
    all_pairs = set()
    for ind, (i, j) in enumerate(pairs):
        if (i, j) in all_pairs:
            continue # discard duplicate pairs
        for k, l in new_true_pairs:
            # keep only pred rels whose objs match true ones, as in IMP
            if i == true_to_pred[k] and j == true_to_pred[l]:
                pred_rel_inds.append(ind)
                all_pairs.add((i, j))
                break
    new_rel_classes = rel_classes.copy()
    set_unspecified(new_rel_classes, pred_rel_inds, 0)
        # set filtered rels to BG

    # Name the kept true objects.
    true_names = [None] * true_obj_classes.shape[0]
        # only name kept (focused and having relationships) true objects
    for i, (a, name) in zip(true_obj_inds,
            classes_to_names(true_obj_classes[true_obj_inds], finalize=False)):
        true_names[i] = (a, name)

    # Name the predicted objects using the names of the kept true objects.
    names = [None] * obj_classes.shape[0]
    for i, j in true_to_pred.items():
        if j is not None:
            names[j] = true_names[i]

    return new_rel_classes, names

#def filter_by_scores(classes, scores, spec):
#    spec = ast.literal_eval(spec)
#    if isinstance(spec, float):
#        mask = threshold_scores(scores, spec, -1) < 0
#    elif isinstance(spec, int):
#        mask = topk_scores(scores, spec, -1) < 0
#    else:
#        raise ValueError('filter value of must be int or float')
#    classes = classes.copy()
#    classes[mask] = 0
#    return classes

def int_tuple_2(s):
    x = ast.literal_eval(s)
    if not isinstance(x, tuple):
        x = (x,)
    assert len(x) == 2
    assert all(isinstance(y, int) for y in x)
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-ground', dest='ground', action='store_false',
            help='use ground truth graph to limit shown objects & relations')
    parser.add_argument('-G', '--truth', action='store_true',
            help='plot ground truth instead of predictions')
    #parser.add_argument('-o', '--obj-filter',
    #        help='threshold or top-k to filter objects')
    #parser.add_argument('-r', '--rel-filter',
    #        help='threshold or top-k to filter relationships')
    #parser.add_argument('-k', '--known-only', action='store_true',
    #        help='show only known objects and their relationships')
    #parser.add_argument('-u', '--unknown-only', action='store_true',
    #        help='show only unknown objects and relationships with unknowns')
    parser.add_argument('-U', '--gt-unknown-only', action='store_true',
            help='show only relationships overlapping with ground-truth unknown'
            ' objects and relationships with unknowns')
    parser.add_argument('--rel-free-objs', action='store_true',
            help='show objects without relationships')
    parser.add_argument('-N', '--gt-based-naming', action='store_true')
    parser.add_argument('--underscore', action='store_true')
    parser.add_argument('--image-relationship', action='store_true')
    parser.add_argument('-m', '--num-images', type=int)
    parser.add_argument('-F', '--format', default='pdf')
    parser.add_argument('--figsize', type=int_tuple_2)
    parser.add_argument('--dpi'    , type=int        )
    parser.add_argument('-O', '--output') # plot output dir
    parser.add_argument('path') # dir with target files to plot
    args = parser.parse_args()
    _, dataset_name, result_dir = analyze_method_dir(args.path)

    # Load dataset info.
    dataset = load_dataset(dataset_name)
    num_obj_classes = len(dataset.ind_to_classes   )
    num_rel_classes = len(dataset.ind_to_predicates)
    imp_cfg.ind_to_class     = dataset.ind_to_classes
    imp_cfg.ind_to_predicate = dataset.ind_to_predicates
        # global configurations required in draw_graph
    if not args.underscore:
        # Remove underscores from special class names.
        imp_cfg.ind_to_class = [
                k.replace('_', '') for k in imp_cfg.ind_to_class]
        imp_cfg.ind_to_predicate = [
                k.replace('_', '') for k in imp_cfg.ind_to_predicate]
    imp_cfg.figsize = args.figsize
    imp_cfg.dpi     = args.dpi

    # Load graph data to plot.
    eval_results = load_eval_results(result_dir)
    visual_info = load(os.path.join(result_dir, 'visual_info.json'))
    indices = list(range(len(visual_info)))
    if args.num_images is not None:
        indices = indices[:args.num_images]

    # Prepare for saving.
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    for i in tqdm(indices): # for each image
        # Load the image.
        filename = visual_info[i]['img_file']
        img = np.asarray(Image.open(filename)) # in RGB

        # Format true data.
        if args.ground or args.truth: # use true obj data corresponding to pred
            # load gt obj & rel data
            truth = eval_results['groundtruths'][i]
            true_boxes = truth.bbox.numpy()
                # Tensor will not be accepted in ground_predictions
            true_obj_classes = truth.get_field('labels'        ).numpy()
            relation_tuple   = truth.get_field('relation_tuple').numpy()
            true_pairs       = relation_tuple[:, :2]
            true_rel_classes = relation_tuple[:,  2]

        # Format predicted data.
        if args.truth: # use true obj and rel data as pred
            # treat the gt data as pred
            boxes = true_boxes
            pairs = true_pairs
            obj_classes = true_obj_classes
            rel_classes = true_rel_classes
            obj_scores = np.ones(true_obj_classes.shape[:1])
            rel_scores = np.ones(true_rel_classes.shape[:1]) # dummy scores
        else:
            # load obj data
            pred = eval_results['predictions'][i]
            boxes = pred.bbox.numpy()
            obj_classes = pred.get_field('pred_labels').numpy()
            obj_scores  = pred.get_field('pred_scores').numpy()

            if 'rel_pair_idxs' in pred.fields(): # relation is on
                # load rel data
                pairs = pred.get_field('rel_pair_idxs').numpy()
                rel_scores  = pred.get_field('pred_rel_scores').numpy()
                rel_classes = pred.get_field('pred_rel_labels').numpy()
            else:
                # make dummy rel data
                pairs      = np.empty((0, 2))
                rel_scores = np.empty((0,  ))
                    # dummy data for setting the number of relations to 0

        if args.ground:
            # Filter out predicted objects not overlapping with truths.
            rel_classes, name_hint = filter_relationships_by_gt(
                    boxes, obj_classes, pairs, rel_classes,
                    true_boxes, true_obj_classes, true_pairs,
                    focus_gt_classes=[dataset.classes_to_ind['__unknown__']]
                        if args.gt_unknown_only else None)

        ## Filter predicted objects by classes.
        #assert not args.unknown_only or not args.known_only
        #if args.unknown_only or args.known_only:
        #    unknown = dataset.classes_to_ind['__unknown__']
        #    obj_inds_1 = (obj_classes == unknown).nonzero()[0]
        #        # unknown objects
        #    rel_inds = (obj_classes[pairs] == unknown).any(1).nonzero()[0]
        #        # relations with any unknown objects
        #    obj_inds_2 = pairs[rel_inds].flatten()
        #        # objects in relations with any unknown objects
        #    if args.unknown_only:
        #        obj_inds = np.union1d(obj_inds_1, obj_inds_2)
        #        set_unspecified(obj_scores, obj_inds, 0)
        #        set_unspecified(rel_scores, rel_inds, 0)
        #    else:
        #        obj_inds = obj_inds_1 # keep knowns related to unknowns
        #        obj_scores[obj_inds] = 0
        #        rel_scores[rel_inds] = 0

        ## Filter predicted objects and relationships by scores.
        #obj_scores[obj_classes == 0] = -1
        #    # prevent selecting BG objs in top-K
        #if args.obj_filter is not None:
        #    obj_classes = filter_by_scores(
        #            obj_classes, obj_scores, args.obj_filter)
        #rel_scores[(obj_classes[pairs] == 0).any(1)] = -1
        #    # prevent selecting rels with BG objs in top-K
        #if args.rel_filter is not None:
        #    rel_scores = filter_by_scores(
        #            rel_classees, rel_scores, args.rel_filter)

        if not args.rel_free_objs:
            # Filter out objects without non-background relationships.
            set_unspecified(obj_classes, np.unique(pairs[rel_classes != 0]), 0)

        # Name the predicted objects.
        names = classes_to_names(obj_classes,
                hint=name_hint if args.gt_based_naming else None)
            # BG objs are separately named from other classes

        # Set output paths.
        if args.output is None:
            imp_cfg.image_filename = imp_cfg.graph_filename = None
        else:
            name, _ = os.path.splitext(os.path.basename(filename))
            imp_cfg.graph_filename = os.path.join(
                    args.output, f'graph-{name}.{args.format}')
            imp_cfg.image_filename = os.path.join(
                    args.output, f'image-{name}.{args.format}')

        # Visualize the predicted data.
        visualize_scene_graph(img, boxes, obj_classes, pairs, rel_classes,
                image_relationship=args.image_relationship, names=names)

if __name__ == '__main__':
    main()
