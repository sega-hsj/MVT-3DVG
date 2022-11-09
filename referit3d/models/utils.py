from unittest import mock
import ipdb
import torch
import torch.nn.functional as F


def my_get_siamese_features(net, in_features, numbers):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    n_scenes,n_items = in_features.shape[:2]
    out_features = []
    for i in range(n_scenes):
        cc=net(in_features[i,:numbers[i]])
        dd=torch.ones(n_items,762).cuda()
        dd[:numbers[i]]=cc
        out_features.append(dd)
    out_features = torch.stack(out_features)
    return out_features

def get_siamese_features(net, in_features, aggregator=None, batch_pnet=False):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    # ipdb.set_trace()
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    if not batch_pnet:
        for i in range(n_items):
            out_features.append(net(in_features[:, i]))

        if aggregator is not None:
            out_features = aggregator(out_features, dim=independent_dim)
    else:
        B,N,K,D = in_features.shape
        out_features = net(in_features.contiguous().view(-1,K,D))
        out_features = out_features.contiguous().view(B,N,out_features.size(-1))


    return out_features


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        if key == 'model':
            if isinstance(value, torch.nn.parallel.DistributedDataParallel):
                checkpoint[key] = value.module.state_dict()
            else:
                checkpoint[key] = value.state_dict()
        else:
            checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
