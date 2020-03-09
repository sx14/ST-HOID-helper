import os
import json
from collections import defaultdict
import numpy as np

from common import voc_ap, viou


def eval_detection_scores(gt_relations, pred_relations, viou_threshold):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx] \
                    and tuple(pred_relation['triplet'][1:]) == tuple(gt_relation['triplet'][1:]):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                             gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                             gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)

                if ov > 0:
                    pred_relation['viou'] = ov
                    pred_relation['hit_gt_id'] = gt_idx

                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx

        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
            pred_relation['viou'] = ov_max
            pred_relation['hit_gt_id'] = k_max
        else:
            if pred_idx < 100:
                pass

    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores, gt_detected


def eval_tagging_scores(gt_relations, pred_relations):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate(groundtruth, prediction_root, zs_cates=None, viou_threshold=0.5,
             det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual
    relation tagging.
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0

    predicate_hit_sum = {}
    sbj_obj_hit_sum = {}
    hit_ranks = []

    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    vid_num = len(groundtruth)
    for v, vid in enumerate(groundtruth.keys()):
        gt_relations = groundtruth[vid]
        print('[%d/%d] %s' % (vid_num, v + 1, vid))
        if len(gt_relations) == 0:
            continue
        tot_gt_relations += len(gt_relations)

        predict_res_path = os.path.join(prediction_root, vid + '.json')
        with open(predict_res_path) as f:
            predict_relations = json.load(f)
            predict_relations = predict_relations[vid]

        if zs_cates is not None:
            zs_predict_relations = []
            for r in predict_relations:
                if tuple(r['triplet']) in zs_cates:
                    zs_predict_relations.append(r)
            predict_relations = zs_predict_relations

        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores, det_gt = eval_detection_scores(
            gt_relations, predict_relations, viou_threshold)

        for i in range(len(det_gt)):
            gt_subject = gt_relations[i]['triplet'][0]
            gt_predicate = gt_relations[i]['triplet'][1]
            gt_object = gt_relations[i]['triplet'][2]

            if gt_predicate in predicate_hit_sum:
                pre_hit_sum = predicate_hit_sum[gt_predicate]
            else:
                pre_hit_sum = {'hit': 0, 'sum': 0}
                predicate_hit_sum[gt_predicate] = pre_hit_sum

            sbj_obj = '%s+%s' % (gt_subject, gt_object)
            if sbj_obj in sbj_obj_hit_sum:
                so_hit_sum = sbj_obj_hit_sum[sbj_obj]
            else:
                so_hit_sum = {'hit': 0, 'sum': 0}
                sbj_obj_hit_sum[sbj_obj] = so_hit_sum

            if det_gt[i] > 0:
                pre_hit_sum['hit'] += 1
                so_hit_sum['hit'] += 1

            pre_hit_sum['sum'] += 1
            so_hit_sum['sum'] += 1

        tp = np.isfinite(det_scores)
        for i in range(len(tp)):
            if tp[i] > 0:
                hit_ranks.append(i)

        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    print('detection mean AP (used in challenge): {}'.format(mean_ap))
    print('detection recall@50: {}'.format(rec_at_n[50]))
    print('detection recall@100: {}'.format(rec_at_n[100]))
    print('tagging precision@1: {}'.format(mprec_at_n[1]))
    print('tagging precision@5: {}'.format(mprec_at_n[5]))
    print('tagging precision@10: {}'.format(mprec_at_n[10]))

    # store scores
    results = []
    results.append('detection mean AP (used in challenge): {}\n'.format(mean_ap))
    results.append('detection recall@50: {}\n'.format(rec_at_n[50]))
    results.append('detection recall@100: {}\n'.format(rec_at_n[100]))
    results.append('tagging precision@1: {}\n'.format(mprec_at_n[1]))
    results.append('tagging precision@5: {}\n'.format(mprec_at_n[5]))
    results.append('tagging precision@10: {}\n'.format(mprec_at_n[10]))
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='relation detection evaluation.')
    parser.add_argument('--dataset', dest='dataset', type=str,
                        default='vidor_hoid_mini')
    args = parser.parse_args()

    gt_path = '../dataset/%s_hoid_test_gt.json' % args.dataset
    print('Loading ground truth from {}'.format(gt_path))
    with open(gt_path, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt)))

    pred_path = '../output/hoid/%s' % args.dataset
    print('Loading prediction from {}'.format(pred_path))

    vid_res_list = os.listdir(pred_path)
    print('Number of videos in prediction: {}'.format(len(vid_res_list)))
    
    eval_results = evaluate(gt, pred_path)
    eval_output_path = '../output/hoid/%s_hoid2_eval_results.txt' % args.dataset
    with open(eval_output_path, 'w') as f:
        f.writelines(eval_results)
