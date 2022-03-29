"""
Utilities to analyze, train, test an 3d_listener.
"""

import torch
import numpy as np
import pandas as pd
import tqdm
import torch.nn.functional as F

from ..utils.evaluation import AverageMeter


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, args, tokenizer=None,epoch=None):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param args:
    :return:
    """

    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)
    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens

        # Forward pass
        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = model(batch, epoch)
        LOSS = LOSS.mean()

        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS
        # Backward
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    return metrics


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, args, randomize=False, tokenizer=None):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.eval()

    if randomize:
        np.random.seed()
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens

        # Forward pass
        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = model(batch)
        LOSS = LOSS.mean()
        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    return metrics


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True,tokenizer=None):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k],list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch['lang_tokens'] = lang_tokens

        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = model(batch)
        LOSS = LOSS.mean()
        out = {}
        out['logits'] = LOGITS
        out['class_logits'] = CLASS_LOGITS
        out['lang_logits'] = LANG_LOGITS

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])

        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])
    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=2020):
    """
    Return the predictions along with the scan data for further visualization
    """
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)

    for batch in data_loader:
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                if isinstance(batch[k],list):
                    continue
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass
        res = model(batch)

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'is_easy': batch['is_easy'][i]
            })

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
