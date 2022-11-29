import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
import time
import datetime
import pickle
import random
from . import constants

from scipy.stats import sem

def formatted_time(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    format_string = '%Y_%m_%d_%Hh%Mm%Ss'
    formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime(format_string)
    return formatted_time

def save_results(path, results):
    pickle.dump(results, open(path+"/results.pickle", "wb"))

def N_scale_to_N(N_scale):
    return {k:int(v*N_scale) for k,v in constants.NEURON_COUNTS().items()}

def N_to_N_scale(N):
    counts = neuron_counts()
    return np.mean([v/counts[k] for k,v in N.items() if k!="SI"])

def flatten_dict(d):
    new_d = {}
    for k,v in d.items():
        if type(v) is dict:
            for k_v, v_v in v.items():
                new_d["_".join([k_v,k])] = v_v
        else:
            new_d[k] = v

    return new_d

def evaluate_performance_pilly(prediction, truth, loss_fn, xp=None):
    mse = loss_fn(prediction[:,2*24:3*24], truth[:,2*24:3*24])
    rmse = torch.sqrt(mse).detach().cpu().numpy()
    return 100*max(.5, ((1-rmse)**10)/(0.5**10+(1-rmse)**10)), None

def make_fig(fig, ax, dir_name="./", name=None, show=False, save_png=True, save_pdf=True, close=True):
    if name is None:
        name = "_".join([str(ax.xaxis.get_label().get_text()), str(ax.yaxis.get_label().get_text())])
    dir_name = dir_name if dir_name[-1] == "/" else dir_name+"/"
    if save_pdf:
        print("Saving pdf figure to", dir_name+name+".pdf")
        fig.savefig(dir_name+name+".pdf", transparent=True, bbox_inches='tight', pad_inches=.01)
    if save_png:
        print("Saving png figure to", dir_name+name+".png")
        fig.savefig(dir_name+name+".png", transparent=False, bbox_inches='tight', pad_inches=.01)
    if show:
        plt.show()
    if close:
        plt.close()

def unique(a,k): # like np.unique with special sorting of mechanisms and target labels

        if k in ["mechanisms", "target_labels"]:
            unique_elements = []
            for element in a:
                if element not in unique_elements:
                    unique_elements.append(element)
            if k=="mechanisms":
                unique_elements = sorted(unique_elements, key=lambda x: (x is not None, x))
            else:
                unique_elements = sorted(unique_elements, key=lambda x: (x is not None, list(READABLE_TARGET_LABEL.keys()).index(tuple(x)) if x is not None else 0))
            return unique_elements
        else:
            return list(np.unique(a))

def muscimol_scheduler(n_sessions, injection_session=None, deinjection_session=None):
    muscimols = np.zeros(n_sessions)

    if deinjection_session is None:
        deinjection_session = n_sessions

    if injection_session is not None:
        muscimols[injection_session:deinjection_session] = 1

    return muscimol

def evaluate_performance_choice(prediction, truth, loss_fn, rng, xp=None):
    # prediction = torch.rand(truth.shape)
    correct_idx = 0 if torch.allclose(truth[:,:24], truth[:,2*24:3*24]) else 1 if torch.allclose(truth[:,24:2*24], truth[:,2*24:3*24]) else None
    correct_loss = loss_fn(truth[:,correct_idx*24:(correct_idx+1)*24], prediction[:,2*24:3*24])
    incorrect_loss = loss_fn(truth[:,(1-correct_idx)*24:(1-correct_idx+1)*24], prediction[:,2*24:3*24])

    if correct_loss < incorrect_loss:
        correct = 1
    elif correct_loss > incorrect_loss:
        correct = 0
    else:
        correct = rng.integers(2)

    if xp is not None:
        perseverative = -1 if correct else xp.check_perseverative(truth[:,2*24:3*24])
    else:
        perseverative = None

    return 100*int(correct), perseverative

evaluate_performance = evaluate_performance_choice
