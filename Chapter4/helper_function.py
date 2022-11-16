from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataset import random_split
plt.style.use('fivethirtyeight')

parent_folder_path = os.path.abspath(os.path.dirname(__file__))


def gen_img(start, target, fill=1, img_size=10):
    # Generates empty image
    img = np.zeros((img_size, img_size), dtype=np.float)

    start_row, start_col = None, None

    if start > 0:
        start_row = start
    else:
        start_col = np.abs(start)

    if target == 0:
        if start_row is None:
            img[:, start_col] = fill
        else:
            img[start_row, :] = fill
    else:
        if start_col == 0:
            start_col = 1

        if target == 1:
            if start_row is not None:
                up = (
                    range(start_row, -1, -1),
                    range(0, start_row + 1),
                )
            else:
                up = (
                    range(img_size - 1, start_col - 1, -1),
                    range(start_col, img_size),
                )
            img[up] = fill
        else:
            if start_row is not None:
                down = (
                    range(start_row, img_size, 1),
                    range(0, img_size - start_row),
                )
            else:
                down = (
                    range(0, img_size - 1 - start_col + 1),
                    range(start_col, img_size),
                )
            img[down] = fill

    return 255 * img.reshape(1, img_size, img_size)


def generate_dataset(img_size=10, n_images=100, binary=True, seed=17):
    np.random.seed(seed)

    starts = np.random.randint(-(img_size - 1), img_size, size=(n_images,))
    targets = np.random.randint(0, 3, size=(n_images,))

    images = np.array(
        [
            gen_img(s, t, img_size=img_size)
            for s, t in zip(starts, targets)
        ], dtype=np.uint8,
    )

    if binary:
        targets = (targets > 0).astype(np.int)

    return images, targets


def plot_images(images, targets, n_plot=30):
    n_rows = n_plot // 6 + ((n_plot % 6) > 0)
    fig, axes = plt.subplots(n_rows, 6, figsize=(9, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):  # noqa
        row, col = i // 6, i % 6
        ax = axes[row, col]
        ax.set_title(f'#{i} - Label:{target}', {'size': 12})
        # plot filter channel in grayscale
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    fig.savefig(os.path.join(parent_folder_path, 'plot_images.png'))


def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier*splits_tensor).long()
    diff = n-splits_tensor.sum()
    splits_tensor[0] += diff
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)


def make_balanced_sampler(y):
    classes, counts = y.unique(return_counts=True)
    weights = 1.0/counts.float()
    sample_weights = weights[y.squeeze().long()]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(
            sample_weights,
        ), generator=generator, replacement=True,
    )
    return sampler
