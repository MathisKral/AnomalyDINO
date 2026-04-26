"""
Temporary file for training memory bank and performing inference.
"""

import faiss
import numpy as np
import os
import tqdm
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import torch

from tqdm import tqdm

from src.model.utils import augment_image
from src.model.backbones import DINOv2Wrapper


def fill_memory_bank(
        model: DINOv2Wrapper,
        data_pth,
        rotation = True,
        knn_metric = "L2_normalized",
        faiss_on_cpu = True,
        masking = False
) -> faiss.IndexFlatL2: 
    """
    Create and fill the memory bank with the given data

    Parameters:
    - model: The backbone model for feature extraction (and, in case of DINOv2, masking).
    - data_pth: The root directory of the dataset.
    - rotation: Whether to augment reference samples with rotation.
    - knn_metric: The metric to use for kNN search. Default is 'L2_normalized' (1 - cosine similarity)
    - faiss_on_cpu: True if the nearest neighbor search should be performed on the cpu instead of the gpu
    - masking: True if the embedding is used to calculate a mask for discarding background patches
    """
    
    assert knn_metric in ["L2", "L2_normalized"]

    img_pths = os.listdir(data_pth)
    with torch.inference_mode():
        features = []

        for img_pth in tqdm(img_pths, desc="Building Memory Bank", leave=False):
            img = cv.cvtColor(cv.imread(os.path.join(data_pth, img_pth)), cv.COLOR_BGR2RGB)

            # augment by rotation if wanted
            if rotation:
                imgs_augmented = augment_image(img)
            else:
                imgs_augmented = [img]

            # calculate features
            for img_aug in imgs_augmented:
                img_tensor, grid_size = model.prepare_image(img_aug)
                img_features = model.extract_features(img_tensor)

                # compute background mask to be able to discard background patches
                img_mask = model.compute_background_mask(img_features, grid_size, threshold=10, masking_type=masking)
                features.append(img_features[img_mask])

        features = np.concatenate(features, axis=0).astype(np.float32)

        # setup faiss nearest neighbor search
        if faiss_on_cpu:
            knn_index = faiss.IndexFlatL2(features.shape[1])
        else:
            resurce = faiss.StandardGpuResources()
            knn_index = faiss.GpuIndexFlatL2(resurce, features.shape[1])

        if knn_metric == "L2_normalized":
            faiss.normalize_L2(features)
        knn_index.add(features)

    return knn_index



def infer(
        img: np.ndarray,
        model: DINOv2Wrapper,
        memory_bank: faiss.IndexFlatL2,
        knn_neighbors,
        knn_metric = "L2_normalized",
        masking = False
) -> np.ndarray:
    
    assert knn_metric in ["L2", "L2_normalized"]
    
    # convert image to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # extract features
    img_tensor, grid_size = model.prepare_image(img)
    img_features = model.extract_features(img_tensor)

    # computing background mask and discard unused features
    mask = model.compute_background_mask(img_features, grid_size, threshold=10, masking_type=masking)
    img_features = img_features[mask]

    # knn calculation
    if knn_metric == "L2":
        distances, _ = memory_bank.search(img_features, k=knn_neighbors)
        if knn_neighbors > 1:
            distances = distances.mean(axis=1)
        distances = np.sqrt(distances)
    else:
        faiss.normalize_L2(img_features)
        distances, _ = memory_bank.search(img_features, k=knn_neighbors)
        if knn_neighbors > 1:
            distances = distances.mean(axis=1)
        distances = distances / 2 # cosine distance

    output_distances = np.zeros_like(mask, dtype=np.float64)
    output_distances[mask] = distances.squeeze()
    output_distances = output_distances.reshape(grid_size)

    return output_distances

def plot_distances(
        input_img: np.ndarray,
        distances: np.ndarray,
        save = ""
) -> matplotlib.figure.Figure:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # plot test image
    ax1.imshow(cv.cvtColor(input_img, cv.COLOR_BGR2RGB))

    # plot distance-map
    plt.colorbar(ax2.imshow(distances), ax=ax2, orientation="vertical")

    ax1.axis("off")
    ax2.axis("off")

    ax1.title.set_text("Test Image")
    ax2.title.set_text("Patch Distances")

    if save != "":
        fig.savefig(save)

    return fig