from sklearn.manifold import TSNE
from matplotlib import cm
from pylab import plt
import torch
import torch.nn as nn


def plot_with_labels(lowDWeights, labels, i):
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 / 9 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer')

    plt.savefig("{}.jpg".format(i))


def visualise_feat(tsne, feat, lbl, epoch):
    # 降维后的数据
    low_dim_embs = tsne.fit_transform(feat.data.numpy())
    # 标签
    labels = lbl.numpy()
    plot_with_labels(low_dim_embs, labels, "epoch{}".format(epoch))


if __name__ == '__main__':
    feat = nn.functional.normalize(torch.rand(8, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(8,))

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)  # TSNE降维，降到2
    visualise_feat(tsne, feat, lbl, 99)
