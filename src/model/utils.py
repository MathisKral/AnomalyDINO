import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

def augment_image(img_ref, augmentation = "rotate", angles = [0, 45, 90, 135, 180, 225, 270, 315]):
    """
    Simply augmentation of images, currently just rotation.
    """
    imgs = []
    if augmentation == "rotate":
        for angle in angles:
            imgs.append(rotate_image(img_ref, angle))
    return imgs


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)
    return result


def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def resize_mask_img(mask, image_shape, grid_size1):
    mask = mask.reshape(grid_size1)
    imgd1 = image_shape[0] // grid_size1[0]
    imgd2 = image_shape[1] // grid_size1[1]
    mask = np.repeat(mask, imgd1, axis=0)
    mask = np.repeat(mask, imgd2, axis=1)
    return mask





