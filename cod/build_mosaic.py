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
    a = min(height, width) // 2
    # center coordinates
    x0, y0 = width // 2, height // 2
    print(height, width)
    print(x0, y0)
    A = (x0 + a, y0)
    B = (x0 + a // 2, y0 + (sqrt(3) * a) // 2)
    C = (x0 - a // 2, y0 + (sqrt(3) * a) // 2)
    D = (x0 - a, y0)
    E = (x0 - a // 2, y0 - (sqrt(3) * a) // 2)
    F = (x0 + a // 2, y0 - (sqrt(3) * a) // 2)
    vertices = np.array([[A, B, C, D, E, F]], dtype=np.int32)
    return vertices


def extract_hexagon(params: Parameters, img):
    return img[params.hexagon_mask]


def replace_hexagon(params, IMG, img, start_x, start_y):
    IMG[np.add(params.hexagon_mask, [start_x, start_y])] = img[params.hexagon_mask]
    return IMG


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

    height, width = images[0].shape[:2]
    vertices = get_vertices_coordinates(height, width)
    mask = np.zeros(images[0].shape, dtype=np.uint8)
    channel_count = images[0].shape[2]
    ignore_mask_color = (255,) * channel_count
    hex_ = cv.fillConvexPoly(mask, vertices, ignore_mask_color)
    hex_mask = []
    for i in range(hex_.shape[0]):
        for j in range(hex_.shape[1]):
            if hex_[i][j] == [255, 255, 255]:
                hex_mask.append([i, j])

    params.hexagon_mask = np.array(hex_mask)

    images = np.array(images)
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

    params.small_images_mean_color = np.mean(images, axis=(1, 2))
    params.small_images = images
    # print(params.small_images_mean_color[0])
    # print(params.small_images_mean_color.shape)
    # print(params.small_images.shape)
    # exit(0)


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    # image.shape -> height, width, channels
    small_image_height, small_image_width = params.small_images[0].shape[:2]
    # new_w = small_image_width * params.num_pieces_horizontal
    # new_h = int((params.image.shape[0] * new_w) / params.image.shape[1]) -
    print(small_image_height)
    print(small_image_width)
    # params.num_pieces_vertical = int((params.image.shape[0] * params.num_pieces_horizontal) / params.image.shape[1])
    params.num_pieces_vertical = int(
        params.image.shape[0] * small_image_width * params.num_pieces_horizontal / params.image.shape[1] /
        params.small_images[0].shape[0])
    # redimensioneaza imaginea
    new_h = params.small_images[0].shape[0] * params.num_pieces_vertical
    new_w = params.small_images[0].shape[1] * params.num_pieces_horizontal
    print(new_h)
    print(new_w)
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
