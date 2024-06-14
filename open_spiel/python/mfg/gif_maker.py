import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from gif_maker import *

plt.rcParams["animation.ffmpeg_path"] = r"/usr/bin/ffmpeg"

def multi_render(datas, filename, labels, vmin=None, vmax=None, cmap='viridis'):

    n_datas = len(datas)

    fig, axes = plt.subplots(1, n_datas+1, figsize = (12, 6))
    ims = []
    if not vmin or not vmax:
        vmax = np.nanmax(datas)
        vmin = np.nanmin(datas)
    for t in range(len(datas[0])):
        imt = []
        for i in range(n_datas):
            axes[i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            im = axes[i].imshow(datas[i][t], animated=True, vmin=vmin, vmax=vmax, cmap=cmap) 
            if t==0 and i==n_datas-1:
                axes[n_datas].axis('off')
                fig.colorbar(im, ax=axes[n_datas])
            imt.append(im)
        ims.append(imt)
        #ims += [[axes[i].imshow(datas[i][t], animated=True, cmap=cmap) for i in range(n_datas)]]
    ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)
    for i in range(n_datas):
        axes[i].set_title(labels[i])
    path = filename
    ani.save(path, writer="ffmpeg", fps=5)
    plt.close()
    print(f"Save {path}")

    fig, axes = plt.subplots(1, n_datas+1, figsize = (12, 6))
    ims = []
    for t in range(len(datas[0])):
        imt = []
        for i in range(n_datas):
            axes[i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            vmin = np.nanmin(datas[i][t])
            vmax = np.nanmax(datas[i][t])
            if np.abs(vmin)>np.abs(vmax):
                vmax = np.abs(vmin)
            else:
                vmin = -np.abs(vmax)
            im = axes[i].imshow(datas[i][t], animated=True, vmin=vmin, vmax=vmax, cmap=cmap) 
            if t==0 and i==n_datas-1:
                axes[n_datas].axis('off')
                #fig.colorbar(im, ax=axes[n_datas])
            imt.append(im)
        ims.append(imt)
        #ims += [[axes[i].imshow(datas[i][t], animated=True, cmap=cmap) for i in range(n_datas)]]
    ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)
    for i in range(n_datas):
        axes[i].set_title(labels[i])
    path =filename[:-4] + 'vimin-max' + filename[-4:]
    ani.save(path, writer="ffmpeg", fps=5)
    plt.close()
    print(f"Save {path}")

class GifMaker():
    def __init__(self):
        self.datas = []

    def add_data(self, d):
        self.datas += [d]

    def add_datas(self, ds):
        self.datas += ds

    def make(self, filename, titles, max_value=None, min_value=None, show=False, cmap='viridis'):
        if not max_value or not min_value:
            datas_np = np.array(self.datas)
            max_value = np.nanmax(datas_np)
            min_value = np.nanmin(datas_np)
        print(f'max: {max_value}')
        print(f'min: {min_value}')

        n_datas = len(self.datas)
        def make_heatmap(i):
            for n in range(n_datas):
                axes[n].cla()
                axes[n].axis('off')
                if titles:
                    ax.set_title(titles[n])
                else:
                    ax.set_title("Time="+str(i))
                data = np.array(self.datas[n][i])
                if n==n_datas-1:
                    sns.heatmap(data, ax=axes[n], cbar=True, cbar_ax=cbar_ax, vmin=min_value, vmax=max_value, cmap=cmap)
                else:
                    sns.heatmap(data, ax=axes[n], vmin=min_value, vmax=max_value, cmap=cmap)
                axes[n].set_aspect('equal', adjustable='box')
        #fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        fms = len(self.datas) 
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, axes = plt.subplots(1, n_datas+1, gridspec_kw = grid_kws, figsize = (12, 8))
        cbar_ax = axes[-1]
        ani = animation.FuncAnimation(fig=fig, func=make_heatmap, frames=fms, interval=500, blit=False)
        ani.save(filename, writer="pillow")

        if show:
            plt.show() 
        plt.close()

    def reset(self):
        plt.close()
        self.datas = []
