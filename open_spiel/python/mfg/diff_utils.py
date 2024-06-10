import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import pyspiel

import copy
from gif_maker import *
from scipy.spatial import distance
from scipy.stats import spearmanr
from scipy.special import kl_div

def distances_plot(data1, data2, path, filename):
    assert len(data1)==3 and len(data2)==3 # (num_agent, num_data, data_vec_size)

    for i in range(num_agent):
        all_cos_sim = []
        all_spearmanr = []
        all_kl_div = []
        all_euclid = []
        for t in range(horizon):
            a = data1[i][t]
            b = data2[i][t]
            cos_sim = 1-distance.cosine(a, b)
            corr, p_value = spearmanr(a, b)
            epsilon = 0.0000001
            kl_div = np.sum([ai * np.log(ai / bi) for ai, bi in zip(a+epsilon, b+epsilon)]) 
            euclid = np.sqrt(np.sum((a-b)**2))
            all_cos_sim.append(cos_sim)
            all_spearmanr.append(corr)
            all_kl_div.append(kl_div)
            all_euclid.append(euclid)

        plt.rcParams["font.size"] = 16 
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.plot(np.arange(len(all_cos_sim)), all_cos_sim, label='cos sim')
        im = ax.plot(np.arange(len(all_spearmanr)), all_spearmanr, label='spearmanr')
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"Corr")
        ax.legend()
        save_path = os.path.join(path, f'corr-{filename}-{i}.png')
        plt.savefig(save_path)
        plt.close()

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.plot(np.arange(len(all_kl_div)), all_kl_div, label='kl divergence')
        im = ax.plot(np.arange(len(all_euclid)), all_euclid, label='euclid')
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"Corr")
        ax.legend()
        save_path = os.path.join(path, f'distance-{filename}-{i}.png')
        plt.savefig(save_path)
        plt.close()

        print(f'saved {save_path} ')
        print(f'mean cos_sim: {np.mean(all_cos_sim)}')

