from src.model.run import fill_memory_bank, infer, plot_distances, save_memory_bank, load_memory_bank
from src.model.backbones import get_model

import cv2 as cv
import os

# configuration ===================================

DATA_PTH = "dataset/custom/paper/train/good"
MODEL_PTH = "models/memory_bank"
OUT_PTH = "out/paper"

KNN_NEIGHBORS = 1

test_img_bad = "dataset/custom/paper/test/bad/bad0.jpeg"
test_img_good = "dataset/custom/paper/test/good/good5.jpeg"

# =================================================


# get model
model = get_model("dinov2_vits14", "cuda", smaller_edge_size=720)

#memory_bank = load_memory_bank(os.path.join(MODEL_PTH, "test.faiss"))

# fill memory bank
memory_bank = fill_memory_bank(
    model,
    DATA_PTH,
)

save_memory_bank(memory_bank, os.path.join(MODEL_PTH, "test.faiss"))


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