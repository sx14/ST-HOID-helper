import os
import glob
from tqdm import tqdm

from dataset import DatasetV1


class VidOR_HOID(DatasetV1):
    """
    VidOR-HOID dataset
    """

    def __init__(self, anno_rpath, video_rpath, splits, low_memory=True):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        low_memory: if true, do not load memory-costly part 
                    of annotations (trajectories) into memory
        """
        super(VidOR_HOID, self).__init__(anno_rpath, video_rpath, splits, low_memory)
        print('VidOR-HOID dataset loaded. {}'.format('(low memory mode enabled)' if low_memory else ''))

    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.anno_rpath, '{}/*/*.json'.format(split)))
        assert len(anno_files)>0, 'No annotation file found for \'{}\'. Please check if the directory is correct.'.format(split)
        return anno_files

    def get_video_path(self, vid):
        return os.path.join(self.video_rpath, self.annos[vid]['video_path'])


if __name__ == '__main__':
    """
    To generate a single JSON groundtruth file for specific split and task,
    run this script from current directory.
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Generate a single JSON groundtruth file for VidOR-HOID')
    parser.add_argument('--data_root', dest='data_root', type=str,
                        default='../data/vidor_hoid_mini/vidor-dataset',
                        help='root dir of dataset')
    parser.add_argument('--split', dest='split', choices=['training', 'validation'],
                        default='validation',
                        help='which dataset split the groundtruth generated for')
    parser.add_argument('--task', dest='task', choices=['obj', 'hoid', 'vrd'],
                        default='hoid',
                        help='which task the groundtruth generated for')
    args = parser.parse_args()

    # to load the trainning set without low memory mode for faster processing, you need sufficient large RAM
    anno_root = os.path.join(args.data_root, 'annotation')
    video_root = os.path.join(args.data_root, 'video')
    dataset = VidOR_HOID(anno_root, video_root, ['validation'], low_memory=True)
    index = dataset.get_index(args.split)

    output_path = 'vidor_hoid_%s_%s_gt.json' % (args.task, args.split)
    print('Generating %s ...' % output_path)
    gts = dict()
    for ind in tqdm(index):
        if args.task == 'obj':
            gt = dataset.get_object_insts(ind)
        elif args.task == 'vrd':
            gt = dataset.get_relation_insts(ind)
        elif args.task == 'hoid':
            gt = dataset.get_relation_insts(ind)
        gts[ind] = gt

    output_path = 'vidor_hoid_mini_%s_%s_gt.json' % (args.task, args.split)
    with open(output_path, 'w') as fout:
        json.dump(gts, fout, separators=(',', ':'))
