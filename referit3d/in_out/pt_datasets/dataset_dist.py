import numpy as np
from torch.utils.data import Dataset
from functools import partial
import warnings
import numpy as np
import torch
import random
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch.distributed as dist
import ipdb
# from .utils import dataset_to_dataloader, max_io_workers

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform, rotate_points_along_z
from ...data_generation.nr3d import decode_stimulus_string
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DistributedSampler as _DistributedSampler

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class ListeningDataset(Dataset):
    def __init__(self, args, training, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False):
        """
        - 增加args, training参数.
        """
        self.training = training 
        self.use_fps = args.use_fps
        self.rank = args.rank
        self.aug_with_pts = args.aug_with_pts
        print("\n [INFO]: create Dataset. training={} | use_fps={} | aug_with_pts={}\n".format(
            self.training,self.use_fps,self.aug_with_pts)) if args.rank==0 else None

        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        # sega_update: 使用原始的token
        #tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        ori_tokens = ref['tokens']
        tokens = " ".join(ori_tokens)
        # tokens = self.vocab(sen).input_ids
        # print(len(tokens))
        # tokens = np.array(tokens)
        # tokens = np.array([102]*(self.max_seq_len + 2 + self.max_context_size * 2))
        # tokens[:min(self.max_seq_len + 2, len(emb))] = emb[:min(self.max_seq_len + 2, len(emb))]
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, is_nr3d, scan_id

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def random_select(self):
        rd_index = np.random.choice(len(self.scans))
        scans_id = list(self.scans.keys())
        rd_objects = self.scans[scans_id[rd_index]].three_d_objects
        obj_id = np.random.choice(len(rd_objects))
        return rd_objects[obj_id]

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d, scan_id = self.get_reference_data(index)
        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        # 增加object points的data augmentation,抑制过拟合.
        samples = np.array([sample_scan_object(o, 
            self.points_per_object, self.training, self.use_fps, self.rank) for o in context])

        # mark their classes
        # res['ori_labels'], 
        # 
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
        res['scan_id'] = scan_id
        box_info = np.zeros((self.max_context_size, 4))
        box_info[:len(context),0] = [o.get_bbox().cx for o in context]
        box_info[:len(context),1] = [o.get_bbox().cy for o in context]
        box_info[:len(context),2] = [o.get_bbox().cz for o in context]
        box_info[:len(context),3] = [o.get_bbox().volume() for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)
        
        # ipdb.set_trace()
        if self.training and self.aug_with_pts:
            obj_nums, pt_nums = samples.shape[:2]
            rd_samples = []
            for idx in range(obj_nums):
                rd_object = self.random_select()
                # ==> (1024,6)
                rd_sample_pt = sample_scan_object(rd_object, self.points_per_object, self.training)
                rd_samples.append(rd_sample_pt)
            rd_samples = np.array(rd_samples)
            if self.object_transformation is not None:
                rd_samples = self.object_transformation(rd_samples)
            select_nums = pt_nums // 3
            for idx in range(obj_nums):
                pt_idx = np.random.choice(pt_nums,select_nums,replace=False)
                samples[idx,pt_idx,:] = rd_samples[idx,pt_idx,:]

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size) # 物体点云.

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]


        res['target_class'] = self.class_to_idx[target.instance_label] 
        res['target_pos'] = target_pos # target位置. 
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['box_info'] = box_info # 物体坐标. 
        res['box_corners'] = box_corners

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size

def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)

def dataset_to_dataloader(args, dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    batch_size_multiplier = 1 if split == 'train' else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == 'train' and len(dataset) % b_size == 1:
        print('dropping last batch during training')
        drop_last = True

    shuffle_ = split == 'train'

    # worker_init_fn = lambda x: np.random.seed(seed)

    if split == 'test':
        if type(seed) is not int:
            warnings.warn('Test split is not seeded in a deterministic manner.')

    if args.dist_train:
        if split == 'train':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None 

    data_loader = DataLoader(dataset,
                             batch_size=b_size,
                             num_workers=n_workers,
                             shuffle=(sampler is None) and shuffle_, # shuffle
                             sampler=sampler,                        # 采样? dist train. 
                             drop_last=drop_last,
                             pin_memory=pin_memory,
                             worker_init_fn=partial(worker_init_fn, seed=seed)
                             )
    return data_loader

def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(
                    args=args,
                    training=(split=='train'),
                    references=d_set,
                    scans=scans,
                    vocab=vocab,
                    max_seq_len=args.max_seq_len,
                    points_per_object=args.points_per_object,
                    max_distractors=max_distractors,
                    class_to_idx=class_to_idx,
                    object_transformation=object_transformation,
                    visualization=args.mode == 'evaluate')

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(args, dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
