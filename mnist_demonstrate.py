import os

import torch, torchvision
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from sklearn.manifold import TSNE

from circle_loss import convert_label_to_similarity, CircleLoss
from visualise_feat import visualise_feat


class ToysModel(nn.Module):
    def __init__(self):
        super(ToysModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x).view(-1, 32)
        x = x / torch.norm(x, dim=1, p=2, keepdim=True)
        return x


def main(resume: bool = True) -> None:
    model = ToysModel().cuda()
    optimizer = Adam(model.parameters(), lr=1e-2)

    train_loader = DataLoader(MNIST(root="./data/", train=True, transform=ToTensor(), download=True), batch_size=64,
                              shuffle=True)
    eval_loader = DataLoader(MNIST(root="./data/", train=False, transform=ToTensor(), download=True), batch_size=2,
                             shuffle=True)

    criterion = CircleLoss(m=0.25, gamma=80)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)  # TSNE降维，降到2

    for epoch in range(20):
        for img, label in tqdm(train_loader):
            img = img.cuda()
            model.zero_grad()
            pred = model(img)
            loss = criterion(*convert_label_to_similarity(pred, label))
            loss.backward()
            optimizer.step()

        visualise_feat(tsne, pred.detach().cpu(), label.detach().cpu(), epoch)

        tp = 1e-10
        fn = 1e-10
        fp = 1e-10
        thresh = 0.75
        for img, label in tqdm(eval_loader):
            img = img.cuda()
            pred = model(img)
            gt_label = label[0] == label[1]
            pred_label = torch.sum(pred[0] * pred[1]) > thresh
            if gt_label and pred_label:
                tp += 1
            elif gt_label and not pred_label:
                fn += 1
            elif not gt_label and pred_label:
                fp += 1

        print("\n")
        print("Recall: {:.4f}".format(tp / (tp + fn)))
        print("Precision: {:.4f}".format(tp / (tp + fp)))

    torch.save(model.state_dict(), "final.pth")


if __name__ == "__main__":
    main()
