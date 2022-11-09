import argparse
import json
import pprint
import os.path as osp
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import torch.distributed as dist
import yaml
import os
import torch
import subprocess
from easydict import EasyDict
from referit3d.utils import str2bool, create_dir
from termcolor import colored
from referit3d.app.init_param import scannet_file,referit3D_file,log_dir,point_tf_ckpt,bert_pretrain_path

cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0



def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))

def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

def init_dist_train(args):
    if args.dist_train:
        # [init dist slurm]
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(22415)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend='nccl')

        total_gpus = dist.get_world_size()
        rank = dist.get_rank()
        args.rank = rank 
        args.total_gpus = total_gpus
    else:
        args.rank = 0 
        args.total_gpus = 1

def parse_arguments(notebook_options=None):
    """Parse the arguments for the training (or test) execution of a ReferIt3D net.
    :param notebook_options: (list) e.g., ['--max-distractors', '100'] to give/parse arguments from inside a jupyter notebook.
    :return:
    """
    
    parser = argparse.ArgumentParser(description='ReferIt3D Nets + Ablations')

    ###################################################################################################
    ########################################### Add Param #############################################
    # Dist Train
    parser.add_argument('--dist_train', type=bool, default=False, help='dist training')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')

    parser.add_argument('--debug_model', type=bool, default=False, help='whether debug in network training')
    parser.add_argument('--batch_pnet', type=bool, default=True, help='batch data for pointnet++ process')
    parser.add_argument('--point_tf_ckpt', type=str, default=point_tf_ckpt, help='ckpt file for point Transformer')

    ## 增加Point Transformer训练
    parser.add_argument('--point_trans', type=bool, default=True, help='Use Point Transformer')
    parser.add_argument('--point_trans_depth', type=int, default=3, help='blocks nums')
    parser.add_argument('--cls_head_finetune', action='store_true', help='model: cls_head_finetune')
    parser.add_argument('--use_fps', action='store_true', help='use FPS for sample points')
    parser.add_argument('--add_color', action='store_true', help='Point Transformer + Color Encoder')
    parser.add_argument('--use_pretraining', type=bool, default=True, help='Use pre-trained Point-BERT')
    parser.add_argument('--aug_with_pts', type=bool, default=False, help='use object points to augment cur sample')


    ####################################################################################################
    #
    # Non-optional arguments
    #
    parser.add_argument('-scannet-file', 
        default=scannet_file,type=str, required=False, help='pkl file containing the data of Scannet'
                                                                       ' as generated by running XXX')
    parser.add_argument('-referit3D-file', 
        default=referit3D_file,
        type=str, required=False)

    #
    # I/O file-related arguments
    #
    parser.add_argument('--log-dir',default=log_dir,type=str, help='where to save training-progress, model, etc')
    parser.add_argument('--resume-path',default=None ,type=str, help='model-path to resume')
    # parser.add_argument('--config-file', type=str, default=None, help='config file')

    #
    # Dataset-oriented arguments
    #
    parser.add_argument('--max-distractors', type=int, default=51,
                        help='Maximum number of distracting objects to be drawn from a scan.')
    parser.add_argument('--max-seq-len', type=int, default=24,
                        help='utterances with more tokens than this they will be ignored.')
    parser.add_argument('--points-per-object', type=int, default=1024,
                        help='points sampled to make a point-cloud per object of a scan.')

    # Note-1 True
    parser.add_argument('--unit-sphere-norm', type=str2bool, default=True,
                        help="Normalize each point-cloud to be in a unit sphere.")

    parser.add_argument('--mentions-target-class-only', type=str2bool, default=True,
                        help='If True, drop references that do not explicitly mention the target-class.')
    parser.add_argument('--min-word-freq', type=int, default=3)
    parser.add_argument('--max-test-objects', type=int, default=88)

    #
    # Training arguments
    #
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'vis'])
    parser.add_argument('--max-train-epochs', type=int, default=100, help='number of training epochs. [default: 100]')
    parser.add_argument('--n-workers', type=int, default=8,
                        help='number of data loading workers [default: -1 is all cores available -1.]')
    parser.add_argument('--random-seed', type=int, default=2020,
                        help='Control pseudo-randomness (net-wise, point-cloud sampling etc.) fostering reproducibility.')
    
    # lr=0.0005 适用于Single GPU; lr=0.001 适用于Multi GPU
    parser.add_argument('--init-lr', type=float, default=0.001, help='learning rate for training.')

    #
    # Model arguments
    #
    parser.add_argument('--model', type=str, default='referIt3DNet_transformer', choices=['referIt3DNet_transformer',])
    parser.add_argument('--bert-pretrain-path', type=str, default=bert_pretrain_path)
    
    parser.add_argument('--view_number', type=int, default=4)
    parser.add_argument('--rotate_number', type=int, default=4)

    parser.add_argument('--label-lang-sup', type=str2bool, default=True)
    parser.add_argument('--aggregate-type', type=str, default='avg')
    
    parser.add_argument('--encoder-layer-num', type=int, default=3)
    parser.add_argument('--decoder-layer-num', type=int, default=4)
    parser.add_argument('--decoder-nhead-num', type=int, default=8)
    
    parser.add_argument('--object-latent-dim', type=int, default=768)
    parser.add_argument('--inner-dim', type=int, default=768)

    parser.add_argument('--dropout-rate', type=float, default=0.15)
    parser.add_argument('--lang-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing the target via '
                                                                          'language only is added.')
    parser.add_argument('--obj-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing for each segmented'
                                                                         ' object its class type is added.')

    #
    # Misc arguments
    #
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device. [default: 0]')
    parser.add_argument('--n-gpus', type=int, default=1, help='number gpu devices. [default: 1]')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size per gpu. [default: 32]')
    parser.add_argument('--save-args', type=str2bool, default=True, help='save arguments in a json.txt')
    parser.add_argument('--experiment-tag', type=str, default=None, help='will be used to name a subdir '
                                                                         'for log-dir if given')
    parser.add_argument('--cluster-pid', type=str, default=None)

    #
    # "Joint" (Sr3d+Nr3D) training
    #
    parser.add_argument('--augment-with-sr3d', type=str, default=None,
                        help='csv with sr3d data to augment training data'
                             'of args.referit3D-file')
    parser.add_argument('--vocab-file', type=str, default=None, help='optional, .pkl file for vocabulary (useful when '
                                                                     'working with multiple dataset and single model.')
    parser.add_argument('--fine-tune', type=str2bool, default=False,
                        help='use if you train with dataset x and then you '
                             'continue training with another dataset')
    parser.add_argument('--s-vs-n-weight', type=float, default=None, help='importance weight of sr3d vs nr3d '
                                                                          'examples [use less than 1]')

    # Parse args
    if notebook_options is not None:
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args()

    if not args.resume_path and not args.log_dir:
        raise ValueError

    init_dist_train(args)
    print(colored("\n************************ START ***************************\n",'green')) if args.rank == 0 else None 

    # if args.config_file is not None:
    #     with open(args.config_file, 'r') as fin:
    #         configs_dict = json.load(fin)
    #         apply_configs(args, configs_dict)

    # Create logging related folders and arguments
    if args.log_dir and (args.rank==0):
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        if args.experiment_tag:
            args.log_dir = osp.join(args.log_dir, args.experiment_tag, timestamp)
        else:
            args.log_dir = osp.join(args.log_dir, timestamp)

        args.checkpoint_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
        args.tensorboard_dir = create_dir(osp.join(args.log_dir, 'tb_logs'))

    if args.resume_path and not args.log_dir and (args.rank==0):  # resume and continue training in previous log-dir.
        checkpoint_dir = osp.split(args.resume_path)[0]  # '/xxx/yyy/log_dir/checkpoints/model.pth'
        args.checkpoint_dir = checkpoint_dir
        args.log_dir = osp.split(checkpoint_dir)[0]
        args.tensorboard_dir = osp.join(args.log_dir, 'tb_logs')

    # Print them nicely
    args_string = pprint.pformat(vars(args))
    print(colored(args_string,'green')) if args.rank == 0 else None 

    if args.save_args and (args.rank==0):
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)
    print(colored('____________________________________________________\n','red')) if args.rank == 0 else None
    return args


def read_saved_args(config_file, override_args=None, verbose=True):
    """
    :param config_file:
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def apply_configs(args, config_dict):
    for k, v in config_dict.items():
        setattr(args, k, v)
