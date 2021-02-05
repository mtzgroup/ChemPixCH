import os
import numpy as np
import os
import cv2
import PIL
from PIL import Image
from multiprocessing import Pool
from .general import run, get_files, delete_file, init_dir

TIMEOUT = 10


def get_max_shape(arrays):
    """
    Args:
        images: list of arrays

    """
    shapes = list(map(lambda x: list(x.shape), arrays))
    ndim = len(arrays[0].shape)
    max_shape = []
    for d in range(ndim):
        max_shape += [max(shapes, key=lambda x: x[d])[d]]

    return max_shape


def pad_batch_images(images, max_shape=None):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad

    """

    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)


def greyscale(state):
    """Preprocess state (:, :, 3) image into greyscale"""
    state = state[:, :, 0]*0.299 + state[:, :, 1]*0.587 + state[:, :, 2]*0.114
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)



def downsample(state):
    """Downsamples an image on the first 2 dimensions

    Args:
        state: (np array) with 3 dimensions

    """
    return state[::2, ::2, :]


def pad_image(img, output_path, pad_size=[8,8,8,8], buckets=None):
    """Pads image with pad size and with buckets

    Args:
        img: (string) path to image
        output_path: (string) path to output image
        pad_size: list of 4 ints
        buckets: ascending ordered list of sizes, [(width, height), ...]

    """
    top, left, bottom, right = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0] + left + right, old_im.size[1] + top + bottom)
    new_size = get_new_size(old_size, buckets)
    new_im = Image.new("RGB", new_size, (255,255,255))
    new_im.paste(old_im, (left, top))
    new_im.save(output_path)


def get_new_size(old_size, buckets):
    """Computes new size from buckets

    Args:
        old_size: (width, height)
        buckets: list of sizes

    Returns:
        new_size: original size or first bucket in iter order that matches the
            size.

    """
    if buckets is None:
        return old_size
    else:
        w, h = old_size
        for (w_b, h_b) in buckets:
            if w_b >= w and h_b >= h:
                return w_b, h_b

        return old_size


def downsample_image(img, output_path, ratio=2):
    """Downsample image by ratio"""
    assert ratio>=1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True


