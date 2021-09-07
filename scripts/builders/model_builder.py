import torchvision.models as models
import os
import torch
import os.path as osp
from MobileNetV1 import mobilenet_w1, get_mobilenet
from collections import OrderedDict
from pprint import pformat

__AVAI_MODELS__ = {
                    'mobilenet_v3_large', 'mobilenet_v1', 'mobilenet_v2',
                  }

def build_model(model_name):
    assert model_name in __AVAI_MODELS__, f"Wrong model name parameter. Expected one of {__AVAI_MODELS__}"

    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained = True)


    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained = True)

    if model_name == 'mobilenet_v1':
        root = os.path.join(os.getenv('MODELS_ROOT'), ".torch", "models") if os.getenv('MODELS_ROOT') else \
            os.path.join("~", ".torch", "models")
        model = get_mobilenet(width_scale=1.0, model_name="mobilenet_w1", root = root, pretrained = True)

    return model

def check_isfile(fpath):
    """Checks if the given path is a file.
    Args:
        fpath (str): file path.
    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers):
    if discarded_layers:
        print(
            '** The following layers are discarded '
            'due to unmatched keys or layer size: {}'.
            format(pformat(discarded_layers))
        )
    if unmatched_layers:
        print(
            '** The following layers were not loaded from checkpoint: {}'.
            format(pformat(unmatched_layers))
        )

def load_pretrained_weights(model, file_path='', pretrained_dict=None, extra_prefix=''):
    r"""Loads pretrianed weights to model. Imported from openvinotoolkit/deep-object-reid.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        file_path (str): path to pretrained weights.
    """
    def _remove_prefix(key, prefix):
        prefix = prefix + '.'
        if key.startswith(prefix):
            key = key[len(prefix):]
        return key

    if file_path:
        check_isfile(file_path)
    checkpoint = (load_checkpoint(file_path)
                       if not pretrained_dict
                       else pretrained_dict)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        k = extra_prefix + _remove_prefix(k, 'module')

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    message = file_path if file_path else "pretrained dict"
    unmatched_layers = sorted(set(model_dict.keys()) - set(new_state_dict))
    if len(matched_layers) == 0:
        print(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually'.format(message)
        )
        _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

        raise RuntimeError(f'The pretrained weights {message} cannot be loaded')
    print(
        'Successfully loaded pretrained weights from "{}"'.
        format(message)
    )
    _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

def load_checkpoint(fpath):
    r"""Loads checkpoint. Imported from openvinotoolkit/deep-object-reid.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint