import os
import json
import argparse

from dataset import VidVRD_HOID, VidOR_HOID
from evaluation import eval_otd, eval_vrd, eval_hoid


def evaluate_interaction(dataset, split, prediction_root):
    # evaluate in default setting
    print('\n\n---------- default -----------')
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)

    cls_pred_root = prediction_root+'_cls'
    print('Processing output files ...')
    eval_hoid.vid_pred_to_cls_pred(groundtruth, prediction_root, cls_pred_root)
    eval_hoid.evaluate(groundtruth, cls_pred_root)

    # evaluate in non-zero-shot setting
    print('\n\n------- non-zero-shot --------')
    nzs_categories = dataset.get_interactions('train')
    groundtruth = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        nzs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet'][1:]) in nzs_categories:
                nzs_gt_relations.append(r)
        if len(nzs_gt_relations) > 0:
            groundtruth[vid] = nzs_gt_relations
    eval_hoid.evaluate(groundtruth, cls_pred_root, nzs_categories)

    # evaluate in zero-shot setting
    print('\n\n--------- zero-shot ----------')
    zs_categories = dataset.get_interactions(split).difference(
            dataset.get_interactions('train'))
    groundtruth = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet'][1:]) in zs_categories:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            groundtruth[vid] = zs_gt_relations
    eval_hoid.evaluate(groundtruth, cls_pred_root, zs_categories)


def evaluate_object(dataset, split, prediction_root):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    eval_otd.evaluate(groundtruth, prediction_root)


def evaluate_relation(dataset, split, prediction_root):
    # evaluate in default setting
    print('==== default ==== ')
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    eval_vrd.evaluate(groundtruth, prediction_root)

    # evaluate in zero-shot setting
    print('==== zero-shot ==== ')
    zeroshot_triplets = dataset.get_triplets(split).difference(
            dataset.get_triplets('train'))
    groundtruth = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet']) in zeroshot_triplets:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            groundtruth[vid] = zs_gt_relations

    eval_vrd.evaluate(groundtruth, prediction_root, zeroshot_triplets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a set of tasks related to spatio-temporal human-object interaction detection.')
    parser.add_argument('--dataset', dest='dataset',type=str,
                        default='vidvrd_hoid',
                        help='the dataset name for evaluation')
    parser.add_argument('--split', dest='split', type=str,
                        default='test',
                        help='the split name for evaluation')
    parser.add_argument('--task', dest='task', choices=['obj', 'hoid', 'vrd'],
                        default='hoid',
                        help='which task to evaluate')
    parser.add_argument('--pred_root', dest='pred_root', type=str,
                        default='output',
                        help='Corresponding prediction JSON file')
    args = parser.parse_args()

    data_dir = os.path.join('data')
    data_root = os.path.join(data_dir, args.dataset)
    anno_root = os.path.join(data_root, 'annotation')
    video_root = os.path.join(data_root, 'video')
    if args.dataset == 'vidvrd_hoid':
        if args.task == 'hoid' or args.task == 'vrd' :
            # load train set for zero-shot evaluation
            dataset = VidVRD_HOID(anno_root, video_root, ['train', args.split])
        else:
            dataset = VidVRD_HOID(anno_root, video_root, [args.split])
    elif args.dataset == 'vidor_hoid':
        if args.task == 'hoid' or args.task == 'vrd' :
            # load train set for zero-shot evaluation
            dataset = VidOR_HOID(anno_root, video_root, ['train', args.split], low_memory=True)
        else:
            dataset = VidOR_HOID(anno_root, video_root, [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    output_path = os.path.join(args.pred_root, args.task, args.dataset)
    print('Loading prediction from {}'.format(output_path))

    vid_res_list = os.listdir(output_path)
    print('Number of videos in prediction: {}'.format(len(vid_res_list)))

    if args.task == 'obj':
        evaluate_object(dataset, args.split, output_path)
    elif args.task == 'vrd':
        evaluate_relation(dataset, args.split, output_path)
    elif args.task == 'hoid':
        evaluate_interaction(dataset, args.split, output_path)
