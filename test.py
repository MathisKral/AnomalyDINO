from src.model.run import fill_memory_bank, infer, plot_distances
from src.model.backbones import get_model

import cv2 as cv
import os

# configuration ===================================

DATA_PTH = "dataset/custom/paper/train/good"
OUT_PTH = "out/paper"

KNN_NEIGHBORS = 1

test_img_bad = "dataset/custom/paper/test/bad/bad0.jpeg"
test_img_good = "dataset/custom/paper/test/good/good5.jpeg"

# =================================================


# get model
model = get_model("dinov2_vits14", "cuda")

# fill memory bank
memory_bank = fill_memory_bank(
    model,
    DATA_PTH,
)

# get test image
img = cv.imread(test_img_good)

# infer distances
distances = infer(
    img,
    model,
    memory_bank,
    KNN_NEIGHBORS
)

# plot result
_ = plot_distances(
    img,
    distances,
    save=os.path.join(OUT_PTH, "test2.jpg")
)