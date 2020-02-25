import os
import glob

from dataset import Dataset


class VidVRD_HOID(Dataset):
    """
    VidVRD-STHOID dataset
    """

    def __init__(self, anno_rpath, video_rpath, splits):
        """
        anno_rpath: the root path of annotations
        video_rpath: the root path of videos
        splits: a list of splits in the dataset to load
        """
        super(VidVRD_HOID, self).__init__(anno_rpath, video_rpath, splits)
        print('VidVRD-HOID dataset loaded.')

    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.anno_rpath, '{}/*.json'.format(split)))
        assert len(anno_files)>0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files

    def get_video_path(self, vid):
        return os.path.join(self.video_rpath, '{}.mp4'.format(vid))


if __name__ == '__main__':
    """
    To generate a single JSON groundtruth file for specific split and task,
    run this script from current directory.
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Generate a single JSON groundtruth file for VidVRD-STHOID')
    parser.add_argument('--data_root', dest='data_root', type=str,
                        default='../data/vidvrd_hoid',
                        help='root dir of dataset')
    parser.add_argument('--split', dest='split', choices=['train', 'test'],
                        default='test',
                        help='which dataset split the groundtruth generated for')
    parser.add_argument('--task', dest='task', choices=['obj', 'hoid', 'vrd'],
                        default='hoid',
                        help='which task the groundtruth generated for')
    args = parser.parse_args()

    # to load the trainning set without low memory mode for faster processing, you need sufficient large RAM
    anno_root = os.path.join(args.data_root, 'annotation')
    video_root = os.path.join(args.data_root, 'video')
    dataset = VidVRD_HOID(anno_root, video_root, ['train', 'test'])
    index = dataset.get_index(args.split)

    gts = dict()
    for ind in index:
        if args.task=='obj':
            gt = dataset.get_object_insts(ind)
        elif args.task=='vrd':
            gt = dataset.get_relation_insts(ind)
        elif args.task == 'hoid':
            gt = dataset.get_relation_insts(ind)
        gts[ind] = gt

    output_path = 'vidvrd_hoid_%s_%s_gt.json' % (args.task, args.split)
    with open(output_path, 'w') as fout:
        json.dump(gts, fout, separators=(',', ':'))
