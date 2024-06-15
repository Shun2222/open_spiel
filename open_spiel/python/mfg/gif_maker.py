import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from gif_maker import *
from sklearn.neighbors import KernelDensity

plt.rcParams["animation.ffmpeg_path"] = r"/usr/bin/ffmpeg"

def multi_render(datas, filename, labels, vmin=None, vmax=None, cmap='viridis', use_kde=False):

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
            if use_kde:
                X, Y, Z, _ = calc_kde(data[i][t])
                im = axes[i].contour(X+0.5, Y+0.5, Z, 10)
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
            if use_kde:
                X, Y, Z, _ = calc_kde(data[i][t])
                im = axes[i].contour(X+0.5, Y+0.5, Z, 10)
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

def calc_kde(prob_datas, num_agent=1000):
    datas = np.array(prob_datas)
    d_shape = n_data.shape
    assert len(d_shape)==2, f'shape error in kde plot {d_shape}'

    n_data = (datas*1000).astype(np.int64)

    # 2次元の確率データの例（仮のデータ）
    #data = np.random.randn(1000, 2)  # 1000個の2次元データポイント
    data = []
    for r in range(d_shape[0]):
        for c in range(d_shape[1]):
            data += [[r, c] for i in range(n_data[r, c])]

    n_data = np.zeros(d_shape)

    # カーネル密度推定を使用してPDFを推定
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)

    # 等高線プロットの準備
    x = y = np.array([i for i in range(d_shape[0])]) 
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    # 確率密度関数の計算
    Z = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)
    Z2 = []
    for i in range(len(Z)):
        Z2.append(Z[len(Z)-i-1])
    Z = Z2
    return X, Y, Z, n_data

    ## 等高線プロット4
    #plt.figure(figsize=(8, 6))
    #plt.imshow(n_data,  extent=[0, d_shape[0], 0, d_shape[1]], alpha=0.3)
    #heatmap = plt.imshow(Z, extent=[0, d_shape[0], 0, d_shape[1]], origin='lower', cmap='viridis', alpha=0.8)
    #plt.contour(X+0.5, Y+0.5, Z, 10)
    #plt.xlabel('X')
    #Jplt.ylabel('Y')
    #plt.title('2D Probability Density Function')
    #plt.legend()
    #plt.show()
