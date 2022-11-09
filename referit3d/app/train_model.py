#!/usr/bin/env python
# coding: utf-8

import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored

# New args
from referit3d.app.configurations import parse_arguments

from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data

# New data loader file
from referit3d.in_out.pt_datasets.dataset_dist import make_data_loaders

from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import ReferIt3DNet_transformer

# New eval mode
from referit3d.models.model_dist import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer, BertModel
from termcolor import colored


def log_train_test_information(epoch):
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{} {}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(epoch, phase,
                                                                        meters[phase + '_total_loss'],
                                                                        meters[phase + '_referential_acc'])

            if args.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            if args.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    # Parse arguments
    args = parse_arguments()
    
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"

    device = torch.device('cuda')
    seed_training_code(args.random_seed+args.rank)
    # torch.backends.cudnn.enabled = True

    # Losses:
    criteria = dict()
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    # Object-type classification
    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

    gpu_num = 1 # len(args.gpu.strip(',').split(','))


    model = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)

    # if gpu_num > 1:
    #     model = nn.DataParallel(model)
    if args.dist_train:
        # sync BN, 否则pointnet++的mean跟var会异常.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    print(colored(model,'green')) if args.rank==0 else None 
    
    # <1>
    if args.point_trans: 
        object_encoder_lr = args.init_lr*0.2
    else:
        object_encoder_lr = args.init_lr

    if gpu_num > 1:
        param_list=[
            {'params':model.module.language_encoder.parameters(),'lr':args.init_lr*0.1},
            {'params':model.module.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
            {'params':model.module.object_encoder.parameters(), 'lr':object_encoder_lr},
            {'params':model.module.obj_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.module.box_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.module.language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.module.object_language_clf.parameters(), 'lr': args.init_lr},
            # add
            {'params':model.cls_head_finetune.parameters(), 'lr':args.init_lr},
        ]
        if not args.label_lang_sup:
            param_list.append( {'params':model.module.obj_clf.parameters(), 'lr': args.init_lr})
    else:
        param_list=[
            {'params':model.language_encoder.parameters(),'lr':args.init_lr*0.1},
            {'params':model.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
            {'params':model.object_encoder.parameters(), 'lr':object_encoder_lr},
            {'params':model.obj_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.box_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.object_language_clf.parameters(), 'lr': args.init_lr},
            # add cls_head_finetune, 使用正常lr.
            {'params':model.cls_head_finetune.parameters(), 'lr':args.init_lr},
        ]
        if not args.label_lang_sup:
            param_list.append( {'params':model.obj_clf.parameters(), 'lr': args.init_lr})

    optimizer = optim.Adam(param_list,lr=args.init_lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[40, 50, 60, 70, 80, 90], gamma=0.65)

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            start_training_epoch = 0
            best_test_epoch = loaded_epoch
            best_test_acc = 0
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir) if args.rank == 0 else None 
        logger = create_logger(args.log_dir) if args.rank == 0 else None
        logger.info(colored('\n **************************** START TRAINING **************************** \n','red')) \
            if args.rank == 0 else None
        
        # dist train
        if args.dist_train:
            model = nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.rank % torch.cuda.device_count()],
                find_unused_parameters=True)
        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                print("cnt_lr", lr_scheduler.get_last_lr()) if args.rank == 0 else None
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args, tokenizer=tokenizer,epoch=epoch)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                # if args.dist_train:
                #     model = nn.parallel.DistributedDataParallel(model, 
                #         device_ids=[args.rank % torch.cuda.device_count()],
                #         broadcast_buffers=False)
                tic = time.time()
                # test_meters = dict({
                #     'test_total_loss': 0,
                #     'test_referential_acc': 0,
                #     'test_object_cls_acc': 0,
                #     'test_txt_cls_acc': 0
                # })
                
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args, tokenizer=tokenizer)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']

                last_test_acc = eval_acc
                last_test_epoch = epoch

                lr_scheduler.step()

                save_state_dicts(osp.join(args.checkpoint_dir, 'last_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler) if args.rank == 0 else None

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green')) if args.rank == 0 else None
                    best_test_acc = eval_acc
                    best_test_epoch = epoch
                    
                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler) if args.rank == 0 else None
                else:
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red')) if args.rank == 0 else None

                log_train_test_information(epoch) if args.rank == 0 else None
                train_meters.update(test_meters)
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                                      main_tag='acc') if args.rank == 0 else None
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                                      step=epoch, main_tag='loss') if args.rank == 0 else None
                if args.rank == 0:
                    bar.refresh()

        if args.rank == 0:
            with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
                f_out.write(('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch)))
                f_out.write(('Last accuracy: {:.4f} (@epoch {})'.format(last_test_acc, last_test_epoch)))

            logger.info('Finished training successfully.')

    elif args.mode == 'evaluate':
        # dist test
        if args.dist_train:
            model = nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.rank % torch.cuda.device_count()],
                broadcast_buffers=False)

        meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args, tokenizer=tokenizer)
        print('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        print('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        print('Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))

        # exit()

        out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                  args, out_file=out_file,tokenizer=tokenizer)
        print(res)
