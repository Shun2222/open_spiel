import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

import numpy as np
import pyspiel

import copy
from gif_maker import *
from scipy.spatial import distance
from scipy.stats import spearmanr
from scipy.special import kl_div

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

def distances_plot(data1, data2, path, filename, xlabel="Time"):
    assert len(data1)==3 and len(data2)==3 # (num_agent, num_data, data_vec_size)

    for i in range(len(data1)):
        all_cos_sim = []
        all_spearmanr = []
        all_kl_div = []
        all_euclid = []
        for t in range(len(data1[0])):
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
        ax.set_xlabel(xlabel)
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

def distances_imshow(data1, data2, path, filename, size=(10, 10), xlabel="State"):
    assert len(data1)==3 and len(data2)==3 # (num_agent, num_data, data_vec_size)

    for i in range(len(data1)):
        all_cos_sim = []
        all_spearmanr = []
        all_kl_div = []
        all_euclid = []
        for t in range(len(data1[0])):
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
        
        plt.rcParams["font.size"] = 12 
        fig = plt.figure(figsize=(48, 12))
        ax = fig.add_subplot(2, 4, 1)
        im = ax.imshow(np.array(all_cos_sim).reshape(size), vmin=-1.0, vmax=1.0, cmap='seismic')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Cos sim")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax = fig.add_subplot(2, 4, 2)
        im = ax.imshow(np.array(all_spearmanr).reshape(size), vmin=-1.0, vmax=1.0, cmap='seismic')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Spearmanr")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax = fig.add_subplot(2, 4, 3)
        im = ax.imshow(np.array(all_kl_div).reshape(size))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"KL divergence")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax = fig.add_subplot(2, 4, 4)
        im = ax.imshow(np.array(all_euclid).reshape(size))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Euclid Distance")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax = fig.add_subplot(2, 4, 5)
        all_cos_sim2 = np.array(all_cos_sim)
        tf = (all_cos_sim2<0.8) & (all_cos_sim2>0)
        all_cos_sim2[tf] = 0
        im = ax.imshow(all_cos_sim2.reshape(size), vmin=-1.0, vmax=1.0, cmap='seismic')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Cos sim")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax = fig.add_subplot(2, 4, 6)
        all_spearmanr2 = np.array(all_spearmanr)
        tf = (all_spearmanr2<0.8) & (all_spearmanr2>0)
        all_spearmanr2[tf] = 0
        im = ax.imshow(all_spearmanr2.reshape(size), vmin=-1.0, vmax=1.0, cmap='seismic')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Spearmanr")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        save_path = os.path.join(path, f'distance-{filename}-{i}.png')
        plt.savefig(save_path)
        plt.close()

        print(f'saved {save_path} ')
        print(f'mean cos_sim: {np.mean(all_cos_sim)}')



def diff_render_distance_plot(datas, pathes, filenames, labels):
    num_agent = len(datas[0]) 

    fig1 = plt.figure(figsize=(16, 12))
    fig2 = plt.figure(figsize=(16, 12))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for p1 in range(len(datas)):
        for p2 in range(p1+1, len(datas)):
            save_path = os.path.join(pathes[p1], f"{filenames[p1]}-{filenames[p2]}.gif")
            diff_data = datas[p1] - datas[p2]
            multi_render(diff_data, save_path, labels, vmin=-1.0, vmax=1.0, cmap='seismic')

            data1 = datas[p1].reshape(3, 40, 100)
            data2 = datas[p2].reshape(3, 40, 100)
            distances_plot(data1, data2, pathes[p1], f'{filenames[p1]}-{filenames[p2]}-time')

            data1 = np.array([data1[n].T for n in range(len(data1))])
            data2 = np.array([data2[n].T for n in range(len(data2))])
            distances_imshow(data1, data2, pathes[p1], f'{filenames[p1]}-{filenames[p2]}-state', xlabel='State')

            if p1==0:
                for i in range(num_agent):
                    all_cos_sim = []
                    all_spearmanr = []
                    all_kl_div = []
                    all_euclid = []
                    for t in range(len(data1[0])):
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

                    ax = fig1.add_subplot(2, num_agent+1, i+1)
                    im = ax.plot(np.arange(len(all_cos_sim)), all_cos_sim, label=f'cos sim:{filenames[p1]}-{filenames[p2]}')
                    im = ax.plot(np.arange(len(all_spearmanr)), all_spearmanr, label='spearmanrm:{filenames[p1]}-{filenames[p2]}', marker='o')
                    if p2==len(datas)-1:
                        plt.rcParams["font.size"] = 16 
                        ax.set_xlabel('State')
                        ax.set_ylabel(r"Corr")
                        ax.legend()
                        save_path = os.path.join(pathes[0], f'corr-{i}-plots.png')
                        plt.savefig(save_path)
                        plt.close()

                    ax = fig2.add_subplot(2, num_agent+1, i+1)
                    im = ax.plot(np.arange(len(all_kl_div)), all_kl_div, label='kl divergence:{filenames[p1]}-{filenames[p2]}')
                    im = ax.plot(np.arange(len(all_euclid)), all_euclid, label='euclid:{filenames[p1]}-{filenames[p2]}', marker='o')
                    if p2==len(datas)-1:
                        plt.rcParams["font.size"] = 16 
                        ax.set_xlabel(r"Time")
                        ax.set_ylabel(r"Corr")
                        ax.legend()
                        save_path = os.path.join(pathes[0], f'distance-{i}-plots.png')
                        plt.savefig(save_path)
                        plt.close()

                    print(f'saved {save_path} ')
                    print(f'mean cos_sim: {np.mean(all_cos_sim)}')
