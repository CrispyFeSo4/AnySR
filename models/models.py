import copy
import torch

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def replace(pretrain_dict, model_dict):
    new_dict = {}
    for k,v in pretrain_dict.items():
        if 'body.0.' in k[15:]:
            newk = k[:15] + k[15:].replace('body.0','conv1')
            new_dict[newk] = v
        elif 'body.2' in k[15:]:
            newk = k[:15] + k[15:].replace('body.2','conv2')
            new_dict[newk] = v
    pretrain_dict.update(new_dict)
    common_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
    model_dict.update(common_dict)
    return model_dict


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']

    model = models[model_spec['name']](**model_args)
    model_dict = model.state_dict()
    if load_sd:
        pretrain_dict = torch.load(model_spec['path'])
        model_dict = replace(pretrain_dict['model']['sd'], model_dict)
        model.load_state_dict(model_dict, strict=False)
    return model

def test_make(model_spec, model_path, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    model_dict = model.state_dict()
    if load_sd:
        path = model_path
        print(f'load data from {path}')
        pretrain_dict = torch.load(path) 
        model.load_state_dict(pretrain_dict['model']['sd'])
    return model