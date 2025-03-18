import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import pandas as pd
import torch
import numpy as np
import random
import os
import yaml

def get_logger(fpath):
    Path(fpath).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name='r')  # set root logger if not set name
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # output to file by using FileHandler
    fh = logging.FileHandler(f"{fpath}/log.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # output to screen by using StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # add Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,title="learning curve",fpath=None,save_fig=True,show_fig=False):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    if save_fig:
        plt.savefig(f"{fpath}/learning_curve.png")
    if show_fig:
        plt.show()

def save_results(res_dic,fpath = None):
    ''' save results
    '''
    Path(fpath).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res_dic)
    df.to_csv(f"{fpath}/res.csv",index=None)

def save_cfgs(cfgs, fpath):
    ''' save config
    '''
    Path(fpath).mkdir(parents=True, exist_ok=True)
 
    with open(f"{fpath}/config.yaml", 'w') as f:
        yaml.dump(cfgs.__dict__, f, default_flow_style=False)

def all_seed(seed = 1):
    ''' omnipotent seed for RL, attention the position of seed function, you'd better put it just following the env create function
    Args:
        env (_type_): 
        seed (int, optional): _description_. Defaults to 1.
    '''
    if seed == 0:
        return
    # print(f"seed = {seed}")
    # env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False