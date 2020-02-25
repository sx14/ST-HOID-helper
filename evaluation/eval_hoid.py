import os
import json
import numpy as np
from tqdm import tqdm

from common import voc_ap, viou_sx

def trajectory_overlap(gt_trajs, pred_traj):
    """
    Calculate overlap among trajectories
    """
    max_overlap = 0
    max_index = 0
    for t, gt_traj in enumerate(gt_trajs):
        s_viou = viou_sx(gt_traj['sub_traj'], gt_traj['duration'], pred_traj['sub_traj'], pred_traj['duration'])
        o_viou = viou_sx(gt_traj['obj_traj'], gt_traj['duration'], pred_traj['obj_traj'], pred_traj['duration'])
        so_viou = min(s_viou, o_viou)

        if so_viou > max_overlap:
            max_overlap = so_viou
            max_index = t

    return max_overlap, max_index


def evaluate(gt, prediction_root, zs_cates=None, use_07_metric=True, thresh_t=0.5):
    """
    Evaluate the predictions
    """
    if zs_cates is None:
        # collect all categories
        gt_classes = set()
        for rlt_insts in gt.values():
            for rlt_inst in rlt_insts:
                gt_classes.add(convert_cate(rlt_inst['triplet'][1:]))
        gt_class_num = len(gt_classes)
    else:
        gt_classes = set()
        for cate in zs_cates:
            gt_classes.add(convert_cate(cate))
        gt_class_num = len(zs_cates)

    ap_class = dict()
    print('Computing average precision AP over {} classes...'.format(gt_class_num))
    for c in gt_classes:

        # load predictions with category c
        cls_pred_file = os.path.join(prediction_root, c+'.json')
        with open(cls_pred_file) as f:
            cls_preds = json.load(f)

        if len(cls_preds) == 0:
            ap_class[c] = 0.
            continue

        npos = 0
        class_recs = {}

        for vid in gt:
            gt_insts = [inst for inst in gt[vid]
                        if convert_cate(inst['triplet'][1:]) == c]
            det = [False] * len(gt_insts)
            npos += len(gt_insts)
            class_recs[vid] = {'gt_insts': gt_insts, 'det': det}

        vids = [inst['vid'] for inst in cls_preds]
        scores = np.array([inst['score'] for inst in cls_preds])

        nd = len(vids)
        fp = np.zeros(nd)
        tp = np.zeros(nd)

        sorted_inds = np.argsort(-scores)
        sorted_vids = [vids[id] for id in sorted_inds]
        sorted_insts = [cls_preds[id] for id in sorted_inds]

        for d in range(nd):
            R = class_recs[sorted_vids[d]]
            gt_insts = R['gt_insts']
            pred_inst = sorted_insts[d]
            max_overlap, max_index = trajectory_overlap(gt_insts, pred_inst)

            if max_overlap >= thresh_t:
                if not R['det'][max_index]:
                    tp[d] = 1.
                    R['det'][max_index] = True
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        ap_class[c] = ap

    # compute mean ap and print
    print('=' * 30)
    ap_class = sorted(ap_class.items(), key=lambda ap_class: ap_class[0])
    total_ap = 0.
    for i, (category, ap) in enumerate(ap_class):
        print('{:>2}{:>20}\t{:.4f}'.format(i + 1, category, ap))
        total_ap += ap
    if gt_class_num > 0:
        mean_ap = total_ap / gt_class_num
    else:
        mean_ap = 0
    print('=' * 30)
    print('{:>22}\t{:.4f}'.format('mean AP', mean_ap))

    return mean_ap, ap_class


def convert_cate(cate):
    cate_str = '+'.join(cate)
    cate_str = cate_str.replace('/', '_')
    return cate_str


def vid_pred_to_cls_pred(gt, vid_pred_root, cls_pred_root):
    if not os.path.exists(cls_pred_root):
        os.makedirs(cls_pred_root)

    # collect all categories
    cates = set()
    for vid in gt:
        insts = gt[vid]
        for inst in insts:
            cate = convert_cate(inst['triplet'][1:])
            cates.add(cate)

    # collect predictions in each category
    for cate in tqdm(cates):
        curr_cls_preds = []
        for vid_file in os.listdir(vid_pred_root):
            vid = vid_file.split('.')[0]
            vid_file_path = os.path.join(vid_pred_root, vid_file)
            with open(vid_file_path) as f:
                vid_insts = json.load(f)
                vid_insts = vid_insts[vid]
                for inst in vid_insts:
                    curr_cate = convert_cate(inst['triplet'][1:])
                    if curr_cate == cate:
                        inst['vid'] = vid
                        curr_cls_preds.append(inst)

        # create a file for predictions with the same category
        cls_pred_path = os.path.join(cls_pred_root, cate+'.json')
        with open(cls_pred_path, 'w') as f:
            json.dump(curr_cls_preds, f)


if __name__ == "__main__":
    """
    You can directly run this script from the parent directory, e.g.,
    python -m evaluation.video_object_detection val_object_groundtruth.json val_object_prediction.json
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(description='object detection evaluation.')
    parser.add_argument('--dataset', dest='dataset', type=str,
                        default='vidvrd_hoid')
    args = parser.parse_args()

    gt_path = '../dataset/%s_hoid_test_gt.json' % args.dataset
    print('Loading ground truth from {}'.format(gt_path))
    with open(gt_path, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt)))

    vid_pred_path = '../output/hoid/%s' % args.dataset
    print('Loading prediction from {}'.format(vid_pred_path))

    vid_res_list = os.listdir(vid_pred_path)
    print('Number of videos in prediction: {}'.format(len(vid_res_list)))

    cls_pred_path = '../output/hoid/%s_cls' % args.dataset
    print('Processing output files ...')
    vid_pred_to_cls_pred(gt, vid_pred_path, cls_pred_path)

    evaluate(gt, cls_pred_path)
