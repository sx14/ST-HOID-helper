import os
import cv2
import json
from tqdm import tqdm


def to_pred(gt, pred_root, data_root):
    print('Generating prediction files with gt ...')
    vid2pid = get_vid2pid(data_root)
    out = {}
    for vid in tqdm(gt):
        frame_path = os.path.join(data_root, vid2pid[vid], vid, '000000.JPEG')
        im = cv2.imread(frame_path)
        im_h, im_w = im.shape[:2]
        insts = gt[vid]
        for inst in insts:
            inst['score'] = 1.0
            inst['start_fid'] = min([int(fid_str) for fid_str in inst['trajectory']])
            inst['end_fid'] = max([int(fid_str) for fid_str in inst['trajectory']])
            inst['height'] = im_h
            inst['width'] = im_w

            traj = inst['trajectory']
            traj_new = {}
            for fid in traj:
                traj_new['%06d' % int(fid)] = traj[fid]
            inst['trajectory'] = traj_new

        out['%s/%s' % (vid2pid[vid], vid)] = insts
    out = {'version': 'VERSION 1.0', 'results': out}
    with open(pred_root, 'w') as f:
        json.dump(out, f)

def get_vid2pid(data_root):
    vid2pid = {}
    for pid in os.listdir(data_root):
        for vid in os.listdir(os.path.join(data_root, pid)):
            vid2pid[vid] = pid
    return vid2pid


dataset = 'vidor_hoid_mini'
task = 'obj'
split = 'val'

gt_path = 'dataset/%s_%s_%s_gt.json' % (dataset, task, split)
pred_path = 'output/%s/%s_%s_%s_pred.json' % (task, dataset, task, split)
data_root = 'data/%s/vidor-ilsvrc/Data/VID/%s' % (dataset, split)

print('Loading %s' % gt_path)
with open(gt_path) as f:
    gt = json.load(f)

to_pred(gt, pred_path, data_root)