from glob import glob
import os.path
import numpy as np
import cv2
import scipy
from scipy import ndimage
import re
import random


def random_shift(img, shift, angle):
    """Translate image in x direction randomly"""
    x_shift = np.random.uniform(-shift, shift)
    # y_shift = np.random.uniform(-shift, shift)
    new_angle = angle + (-x_shift * 0.004)  # 0.004 angle change for every pixel shifted

    return ndimage.shift(img, (0, x_shift, 0)), new_angle


def random_brightness(image):
    """Apply random brightness on an image"""
    # new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    new_img = np.array(image)
    random_bright = .5 + np.random.uniform()
    new_img[:, :, 2] = new_img[:, :, 2] * random_bright
    new_img[:, :, 2][new_img[:, :, 2] > 255] = 255
    new_img = np.array(new_img)

    # new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)

    return new_img


def random_saturation(image):
    """Apply random saturation on HSV formatted image"""
    saturation_threshold = 0.4 + 1.2 * np.random.uniform()
    new_img = np.array(image)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    new_img[:, :, 1] = new_img[:, :, 1] * saturation_threshold
    new_img = np.array(new_img)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img


def random_shadow(image):
    top_y = image.shape[1] * np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1] * np.random.uniform()
    # image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_hls = image
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    # new_img = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    new_img = image_hls

    return new_img


def random_lightness(image):
    lightness_threshold = 0.2 + 1.4 * np.random.uniform()
    new_img = np.array(image)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HLS)
    new_img[:, :, 1] = new_img[:, :, 1] * lightness_threshold
    # new_img = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)

    return new_img


def preprocess_img(new_img, gt_img):
    """Preprocess and augment the image"""

    # new_angle = angle

    # Crop image, 60 pixels from top and 25 from bottom
    # new_img = image[60:135, :, :]

    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    # Resize image to 200 width and 66 height
    # new_img = cv2.resize(new_img, (64, 64), interpolation=cv2.INTER_AREA)

    if flip_a_coin():
        # Apply blurring
        new_img = ndimage.gaussian_filter(new_img, sigma=3)
        # gt_img = ndimage.gaussian_filter(gt_img, sigma=3)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    if flip_a_coin():
        # Apply random brightness
        new_img = random_brightness(new_img)

    if flip_a_coin():
        # Apply random saturation
        new_img = random_saturation(new_img)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HLS)

    if flip_a_coin():
        # Apply random lightness
        new_img = random_lightness(new_img)

    if flip_a_coin():
        # Apply random shadow
        new_img = random_shadow(new_img)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_HLS2RGB)

    if flip_a_coin():
        # Apply random translation in the x-axis
        x_shift = np.random.uniform(-25, 25)
        new_img = ndimage.shift(new_img, (0, x_shift, 0))
        gt_img = ndimage.shift(gt_img, (0, x_shift, 0))

    if flip_a_coin():
        # Flip image and
        new_img = np.fliplr(new_img)
        gt_img = np.fliplr(gt_img)

    return new_img, gt_img


def flip_a_coin():
    return np.random.uniform() <= .5


def run(data_folder):
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
    }

    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]

        image = scipy.misc.imread(image_file)
        gt_image = scipy.misc.imread(gt_image_file)

        image_base_name = os.path.basename(image_file)
        gt_image_base_name = os.path.basename(gt_image_file)
        image_insert_idx = image_base_name.index(".")
        gt_image_insert_idx = gt_image_base_name.index(".")
        for i in range(5):
            new_image_file_name = image_base_name[:image_insert_idx] + "_" + str(i) + image_base_name[image_insert_idx:]
            new_gt_image_file_name = gt_image_base_name[:gt_image_insert_idx] + "_" + str(i) + gt_image_base_name[
                                                                                               gt_image_insert_idx:]
            image, gt_image = preprocess_img(image, gt_image)
            scipy.misc.imsave(os.path.join(data_folder, "image_2", new_image_file_name), image)
            scipy.misc.imsave(os.path.join(data_folder, "gt_image_2", new_gt_image_file_name), gt_image)


if __name__ == '__main__':
    data_dir = './data/data_road/training'
    run(data_dir)
