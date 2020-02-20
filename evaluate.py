import os
import json
import argparse

from dataset import VidVRD_STHOID, VidOR_STHOID
from evaluation import otd_eval, sthoid_eval


def evaluate_object(dataset, split, prediction_root):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = otd_eval.evaluate(groundtruth, prediction_root)


def evaluate_human_object_interaction(dataset, split, prediction_root):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mean_ap, rec_at_n, mprec_at_n = sthoid_eval.evaluate(groundtruth, prediction_root)
    # evaluate in zero-shot setting
    zeroshot_triplets = dataset.get_triplets(split).difference(
            dataset.get_triplets('train'))
    groundtruth = dict()
    zs_prediction = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet']) in zeroshot_triplets:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            groundtruth[vid] = zs_gt_relations
            with open(os.path.join(prediction_root, vid+'.json')) as f:
                prediction = json.load(f)
                prediction = prediction[vid]
            zs_prediction[vid] = []
            for r in prediction.get(vid, []):
                if tuple(r['triplet']) in zeroshot_triplets:
                    zs_prediction[vid].append(r)
    mean_ap, rec_at_n, mprec_at_n = sthoid_eval.evaluate(groundtruth, zs_prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a set of tasks related to spatio-temporal human-object interaction detection.')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation', default='vidor_sthoid')
    parser.add_argument('split', type=str, help='the split name for evaluation', default='test')
    parser.add_argument('task', choices=['obj', 'hoi'], help='which task to evaluate', default='hoi')
    parser.add_argument('prediction', type=str, help='Corresponding prediction JSON file')
    args = parser.parse_args()

    data_dir = os.path.join('..', 'data')
    data_root = os.path.join(data_dir, args.dataset)
    anno_root = os.path.join(data_root, 'annotation')
    video_root = os.path.join(data_root, 'video')
    if args.dataset=='vidvrd_sthoid':
        if args.task=='hoi':
            # load train set for zero-shot evaluation
            dataset = VidVRD_STHOID(anno_root, video_root, ['train', args.split])
        else:
            dataset = VidVRD_STHOID(anno_root, video_root, [args.split])
    elif args.dataset=='vidor_sthoid':
        if args.task=='hoi':
            # load train set for zero-shot evaluation
            dataset = VidOR_STHOID(anno_root, video_root, ['training', args.split], low_memory=True)
        else:
            dataset = VidOR_STHOID(anno_root, video_root, [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    print('Loading prediction from {}'.format(args.prediction))

    vid_res_list = os.listdir(args.prediction)
    print('Number of videos in prediction: {}'.format(len(vid_res_list)))

    if args.task=='obj':
        evaluate_object(dataset, args.split, args.prediction)
    elif args.task=='hoi':
        evaluate_human_object_interaction(dataset, args.split, args.prediction)
