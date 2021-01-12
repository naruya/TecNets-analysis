from collections import OrderedDict
from matplotlib.markers import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import torch
from torch.distributions import Normal

markers = list(OrderedDict([
    ('o', 'circle'),
    ('v', 'triangle_down'),
    ('^', 'triangle_up'),
    ('<', 'triangle_left'),
    ('>', 'triangle_right'),
    ('8', 'octagon'),
    ('s', 'square'),
    ('p', 'pentagon'),
    ('*', 'star'),
    ('D', 'diamond'),
    ('d', 'thin_diamond'),
    (CARETLEFT, 'caretleft'),
    (CARETRIGHT, 'caretright'),
    (CARETUP, 'caretup'),
    (CARETDOWN, 'caretdown'),
]))*100

colors = ["r", "g", "b", "c", "m", "y", "k"] + \
    ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']*100


def vread(path, T=100):
    cap = cv2.VideoCapture(path)
    gif = [cap.read()[1][:,:,::-1] for i in range(T)]
    gif = np.array(gif)
    cap.release()
    return gif


def extract_emb_np(path):

    with open(path, "rb") as f:
        results = pickle.load(f)

    train_means = np.array([np.squeeze(result['sentence']['mean']) \
                           for result in results['train']])
    train_stds = np.array([np.squeeze(result['sentence']['stddev']) \
                          for result in results['train']])
    train_paths = [result['demo_path'] for result in results['train']]
    test_means = np.array([np.squeeze(result['sentence']['mean']) \
                          for result in results['test']])
    test_stds = np.array([np.squeeze(result['sentence']['stddev']) \
                         for result in results['test']])
    test_paths = [result['demo_path'] for result in results['test']]
    
    print("train_means.shape, train_stds.shape")
    print(train_means.shape, train_stds.shape)
    print("test_means.shape, test_stds.shape")
    print(test_means.shape, test_stds.shape)

    means = np.vstack([train_means, test_means])
    stds = np.vstack([train_stds, test_stds])
    paths = train_paths + test_paths
    
    print("means.shape, stds.shape")
    print(means.shape, stds.shape)
    
    return {
        "train_means": train_means,
        "train_stds": train_stds,
        "train_paths": train_paths,
        "test_means": test_means,
        "test_stds": test_stds,
        "test_paths": test_paths,
        "means": means,
        "stds": stds,
        "paths": paths,
    }


def extract_logs_np(path):
    pass


def poe(prior, posterior, z_dim, eps=1e-8):
    vars = torch.cat((prior.variance.view(-1,1,z_dim), posterior.variance.view(-1,1,z_dim)), dim=1) + eps
    var = 1. / torch.sum(torch.reciprocal(vars), dim=1) + eps
    locs = torch.cat((prior.loc.view(-1,1,z_dim), posterior.loc.view(-1,1,z_dim)), dim=1)
    loc = torch.sum(locs / vars, dim=1) * var
    return Normal(loc, torch.sqrt(var))


def make_emb_input(path):
    inp = np.array(vread(path).transpose(0,3,1,2)[[0,-1]].reshape(-1,6,64,64), np.float32)/255.0  # 1,6,64,64
    inp = torch.from_numpy(inp)
    return inp


def plt_base(title, font_size=14, figsize=(4,4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.title(title, fontsize=font_size)
    fig.patch.set_facecolor('white')  # 図全体の背景色
    fig.patch.set_alpha(1)  # 図全体の背景透明度
    ax.patch.set_facecolor('gray')  # subplotの背景色
    ax.patch.set_alpha(0.2)  # subplotの背景透明度
    return fig, ax
