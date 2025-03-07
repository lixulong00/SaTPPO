import random
import time
import numpy as np
import pytz
import datetime
import os
import torch

def seed_everything(seed):
    """
    Set global random seeds to ensure reproducibility of experiments.

    Args:
    seed (int): The value of the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def unique_folder_name():
    """
    Generate a unique folder name based on the current time.

    Returns:
    str: A unique folder name.
    """
    tz = pytz.timezone('Asia/Shanghai')
    cn_time = datetime.datetime.now(tz)
    return f'{cn_time.year}-{cn_time.month}-{cn_time.day}-{cn_time.hour}-{cn_time.minute}-{cn_time.second}'

def save_model(nets, path, save_network_type='model'):
    """
    Save models or network parameters.

    Args:
    nets (dict): Dictionary containing the networks.
    path (str): Path to save the models.
    save_network_type (str): Type of saving, can be 'model' (entire model) or other (only parameters).
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if save_network_type == 'model':
        for key in nets.keys():
            torch.save(nets[key], path + "/" + key + ".pth")
    else:
        for key in nets.keys():
            torch.save(nets[key].state_dict(), path + "/" + key + ".pth")

def save_args(args, path):
    """
    Save training or experiment arguments.

    Args:
    args (object): Object containing the arguments.
    path (str): Path to save the arguments file.
    """
    with open(path+'/args.txt', 'w') as f:
        for k in args.__dict__:
            a = f'{k}: {str(args.__dict__[k])}'
            f.write(a+'\n')

def make_savedir(args):
    """
    Create a directory to save models and results.

    Args:
    args (object): Object containing the arguments, must have `save_path` and `alg` attributes.
    """
    save_folder = unique_folder_name()  
    args.full_save_path = args.save_path + '/' + save_folder + '_' +  args.alg
    if not os.path.exists(args.full_save_path):
        os.makedirs(args.full_save_path)