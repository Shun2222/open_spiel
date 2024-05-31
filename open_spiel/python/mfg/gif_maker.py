import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams["animation.ffmpeg_path"] = r"/usr/bin/ffmpeg"

def multi_render(datas, filename, labels):

    n_datas = len(datas)
    assert n_datas>=len(labels)

    fig, axes = plt.subplots(1, n_datas, figsize = (12, 6))
    ims = []
    for t in range(len(datas[0])):
        ims += [[axes[i].imshow(datas[i][t], animated=True) for i in range(n_datas)]]
    ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)
    for i in range(len(datas)):
        axes[i].axis('off')
        axes[i].set_title(f"{labels[i]}")
    path = filename
    ani.save(path, writer="ffmpeg", fps=5)
    plt.close()
    print(f"Save {path}")
