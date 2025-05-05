from .networks.pose_conformer import get_pose_conformer
from .networks.pose_cnn_bilstm_sa import get_pose_cnn_bilstm_sa
from .networks.pose_mconformer import get_pose_mconformer
from .networks.pose_transformer import get_pose_transformer
from .networks.pose_conv import get_pose_conv
import torch

_model_factory = {
    'conformer': get_pose_conformer,
    'cnnbilstmsa': get_pose_cnn_bilstm_sa,
    'mconformer': get_pose_mconformer,
    'transformer': get_pose_transformer,
    'conv': get_pose_conv,
}


def create_model(arch, heads, head_conv, wavelet_setting=None):
    model_name = arch[:arch.find('_')] if '_' in arch else arch
    model_scale = arch[arch.find('_') + 1:] if '_' in arch else "medium"
    get_model = _model_factory[model_name]
    model = get_model(model_scale=model_scale, heads=heads, head_conv=head_conv, wavelet_setting=wavelet_setting, )
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None, load_dino=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    if load_dino:
        state_dict_ =  checkpoint['student']
    else:
        state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # Model loading
    # TODO: debug
    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. '
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume and not load_dino:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)