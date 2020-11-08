import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from add_pieces_mosaic import *
from parameters import *
from math import sqrt


# https://www.quora.com/How-can-you-find-the-coordinates-in-a-hexagon
# https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region

def get_vertices_coordinates(height, width):
    A = (width - 1, height // 2)
    B = (3 * width // 4, 0)
    C = (width // 4, 0)
    D = (0, height // 2)
    E = (width // 4, height - 1)
    F = (3 * width // 4, height - 1)
    vertices = np.array([[A, B, C, D, E, F]], dtype=np.int32)
    return vertices


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
    filenames = os.listdir(params.small_images_dir)

    # citeste imaginile din director

    images = []
    for filename in filenames:
        image = cv.imread('{}/{}'.format(params.small_images_dir, filename))
        images.append(image)

    images = np.array(images)
    params.small_images_shape = images.shape
    if params.hexagon:
        height, width = images[0].shape[:2]
        vertices = get_vertices_coordinates(height, width)
        mask = np.zeros(images[0].shape, dtype=np.uint8)
        channel_count = images[0].shape[2]
        ignore_mask_color = (255,) * channel_count
        hex_ = cv.fillConvexPoly(mask, vertices, ignore_mask_color)
        r_mask = []
        c_mask = []
        for i in range(hex_.shape[0]):
            for j in range(hex_.shape[1]):
                if (hex_[i][j] == np.array([255, 255, 255])).all():
                    r_mask.append(i)
                    c_mask.append(j)

        new_images = []
        params.hexagon_mask = np.array([r_mask, c_mask])
        for i, image in enumerate(images):
            new_images.append(extract_hexagon(params, image))

        images = np.array(new_images)

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    if params.hexagon:
        params.small_images_mean_color = np.mean(images, axis=1)
    else:params.small_images_mean_color = np.mean(images, axis=(1, 2))
    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    # image.shape -> height, width, channels
    _, small_image_height, small_image_width, _ = params.small_images_shape
    params.num_pieces_vertical = int(
        params.image.shape[0] * small_image_width * params.num_pieces_horizontal / params.image.shape[1] /
        small_image_height)
    # redimensioneaza imaginea
    if params.hexagon:
        new_w = small_image_width * (params.num_pieces_horizontal + 1)
        params.num_pieces_horizontal = int(params.num_pieces_horizontal // 1.5)
    else:
        new_w = small_image_width * params.num_pieces_horizontal

    new_h = small_image_height * params.num_pieces_vertical
    # print(new_w, new_h)
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
