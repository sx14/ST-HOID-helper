import os
import json
from tqdm import tqdm


def to_pred(gt, pred_root):
    print('Generating prediction files with gt ...')
    for vid in tqdm(gt):
        insts = gt[vid]
        for inst in insts:
            inst['score'] = 1.0
            inst.pop('subject_tid')
            inst.pop('object_tid')
        out = {vid: insts}

        out_path = os.path.join(pred_root, vid+'.json')
        with open(out_path, 'w') as f:
            json.dump(out, f)



dataset = 'vidvrd_hoid'
task = 'vrd'
split = 'test'

gt_path = 'dataset/%s_%s_%s_gt.json' % (dataset, task, split)
pred_root = 'output/%s/%s' % (task, dataset)

if not os.path.exists(pred_root):
    os.makedirs(pred_root)

print('Loading %s' % gt_path)
with open(gt_path) as f:
    gt = json.load(f)

to_pred(gt, pred_root)