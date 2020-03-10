import os
import json
from tqdm import tqdm


def to_pred(gt, pred_root):
    print('Generating prediction files with gt ...')
    out = {}
    for vid in tqdm(gt):
        insts = gt[vid]
        for inst in insts:
            inst['score'] = 1.0
            inst['start_fid'] = min(inst['trajectory'].keys())
            inst['end_fid'] = '%06d' % int(max(inst['trajectory'].keys())+1)
        out[vid] = insts
    out = {'version': 'VERSION 1.0', 'results': out}
    with open(pred_root, 'w') as f:
        json.dump(out, f)


dataset = 'vidor_hoid_mini'
task = 'obj'
split = 'val'

gt_path = 'dataset/%s_%s_%s_gt.json' % (dataset, task, split)
pred_path = 'output/%s/%s_%s_%s_pred.json' % (task, dataset, task, split)

print('Loading %s' % gt_path)
with open(gt_path) as f:
    gt = json.load(f)

to_pred(gt, pred_path)