import random
import math
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import convnets
from . import layers
from . import utils

# -----------------------------
# Model definitions
# -----------------------------


class OmniglotSiamese(nn.Module):
    """Convnet for Siamese network used for the Omniglot dataset.

    References:
    [1] Siamese Neural Networks for One-shot Image Recognition
        Koch et al, 2015.
        https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    def __init__(self, type='weighted_l1'):
        super().__init__()

        if type not in ('weighted_l1', 'contrastive_loss'):
            raise ValueError(f'Unknown siamese type: <{type}>.')
        self._type = type

        blocks_config = [
            dict(
                in_channels=1,
                out_channels=64,
                kernel_size=10,
                pooling=True,
                pool_stride=2
            ),
            dict(
                in_channels=64,
                out_channels=128,
                kernel_size=7,
                pooling=True,
                pool_stride=2
            ),
            dict(
                in_channels=128,
                out_channels=128,
                kernel_size=4,
                pooling=True,
                pool_stride=2
            ),
            dict(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                pooling=True,
                pool_stride=2
            )
        ]

        self._convnet = convnets.ConvNet(blocks_config)

        self.fc1 = nn.Linear(4096, 4096, bias=False)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1)


    def _forward_image(self, x):
        y = self._convnet(x)
        y = layers.flatten(y)

        y = self.fc1(y)
        y = torch.sigmoid(self.bn1(y))
        return y

    def forward(self, x1, x2):
        y1 = self._forward_image(x1)
        y2 = self._forward_image(x2)

        if self._type == 'weighted_l1':
            dist = torch.abs(y1 - y2)
            return torch.sigmoid(self.fc2(dist))
        elif self._type == 'contrastive_loss':
            dist = torch.sqrt(torch.sum(torch.pow(y1 - y2, 2)))
            return dist


def contrastive_loss(dist, y_true, margin=1.):
    return torch.mean(
        (1 - y_true) * torch.pow(dist, 2)\
        + y_true * torch.pow(torch.clamp(margin - dist , min=0.), 2)
    )


# -----------------------------
# Loader definitions
# -----------------------------


class SiameseDataset(Dataset):
    def __init__(self, paths, classes, same_prob=0.5):
        self._paths = paths
        self._classes = classes
        self._same_prob = same_prob

        self._prepare_pairs()

        self._target_size = (105, 105)

    def _prepare_pairs(self):
        self._classes_index = np.unique(self._classes)

        self._index = {
            class_id: np.where(self._classes == class_id)[0]
            for class_id in self._classes_index
        }

    @staticmethod
    def _get_dist(class_1, class_2):
        dist = 0 if class_1 == class_2 else 1
        return np.array([dist])

    def __len__(self):
        return len(self._classes)

    def __getitem__(self, idx):
        class_1 = self._classes[idx]
        img_1 = utils.load_image(self._paths[idx], self._target_size)

        if random.random() > self._same_prob: # Pair of same class.
            class_2 = class_1

            idx_2 = np.random.choice(self._index[class_1][self._index[class_1] != idx])
            img_2 = utils.load_image(self._paths[idx_2], self._target_size)
        else: # Pair of different classes.
            class_2 = np.random.choice(self._classes_index[self._classes_index != class_1])

            path = self._paths[np.random.choice(self._index[class_2])]
            img_2 = utils.load_image(path, self._target_size)

        return {
            'img_1': torch.from_numpy(img_1.transpose(2, 0, 1)).float(),
            'img_2': torch.from_numpy(img_2.transpose(2, 0, 1)).float(),
            'dist': torch.from_numpy(self._get_dist(class_1, class_2)).float()
        }


class SiameseTest:
    def __init__(self, paths, classes, n_ways=5, k_shots=1, device='cpu'):
        self._paths = paths
        self._classes = classes
        self._n_ways = n_ways
        self._k_shots = k_shots
        self._device = device

        self._prepare_pairs()

        self._target_size = (105, 105)

    def _prepare_pairs(self):
        self._classes_index = np.unique(self._classes)

        self._index = {
            class_id: np.where(self._classes == class_id)[0]
            for class_id in self._classes_index
        }

    def _test(self, model, return_imgs=False):
        classes = np.random.choice(self._classes_index, size=self._n_ways, replace=False)

        # classes[0] is the True class
        true_class_paths = np.random.choice(self._index[classes[0]], size=2, replace=False)
        test_img = utils.load_image(self._paths[true_class_paths[0]], self._target_size)
        closest_img = utils.load_image(self._paths[true_class_paths[1]], self._target_size)

        dist_1 = model(self._to_torch(test_img), self._to_torch(closest_img))

        if return_imgs:
            imgs = [(closest_img, dist_1.item())]

        for class_id in classes[1:]:
            idxes = np.random.choice(self._index[class_id], size=self._k_shots, replace=False)
            for shot in range(self._k_shots):
                img = utils.load_image(self._paths[idxes[shot]], self._target_size)

                dist_2 = model(self._to_torch(test_img), self._to_torch(img))

                if not return_imgs and dist_2 < dist_1:
                    return 0
                elif return_imgs:
                    imgs.append((img, dist_2.item()))

        if return_imgs:
            return test_img, sorted(imgs, key=itemgetter(1))

        # The same-class image was found to be to the most similar to the test
        # img: We got it right!
        return 1

    def _to_torch(self, img):
        return torch.from_numpy(
            np.expand_dims(img.transpose(2, 0, 1), axis=0)
        ).float().to(self._device)

    def test(self, model, n_trys=300):
        acc = 0
        for _ in range(n_trys):
            acc += self._test(model)
        return acc / n_trys

    def display_try(self, model):
        test_img, imgs = self._test(model, return_imgs=True)

        plt.imshow(test_img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title('Test image')
        plt.pause(0.1)

        for i, (img, dist) in enumerate(imgs, start=1):
            plt.subplot(math.ceil(self._n_ways / 4), 4, i)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title(f'dist = {dist:.2f}')
