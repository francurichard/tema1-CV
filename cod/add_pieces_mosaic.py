from parameters import *
import numpy as np
import pdb
import timeit
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt



def get_best_match(params: Parameters, area_to_match, metric='euclidean'):
    mean_color = np.mean(area_to_match, axis=(0, 1))
    mean_color = mean_color.reshape(1, -1)
    min_, best = float('inf'), -1
    for i, mean_ in enumerate(params.small_images_mean_color):
        mean_ = mean_.reshape(1, -1)
        dist = pairwise_distances(mean_color, mean_, metric=metric)
        if dist < min_:
            min_ = dist
            best = i

    return params.small_images[best]


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        unit_height_rescale = params.image.shape[0] / params.num_pieces_vertical
        unit_width_rescale = params.image.shape[1] / params.num_pieces_horizontal
        print(unit_width_rescale)
        print(unit_height_rescale)
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                area_to_match = \
                    params.image[int(i * unit_height_rescale):int((i + 1) * unit_height_rescale),
                    int(j * unit_width_rescale):int((j + 1) * unit_width_rescale), :]
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = get_best_match(params, area_to_match)
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    y_pos = np.random.choice(h - H, params.num_pieces_vertical, replace=False)
    x_pos = np.random.choice(w - W, params.num_pieces_horizontal, replace=False)
    unit_height_rescale = params.image.shape[0] / params.num_pieces_vertical
    unit_width_rescale = params.image.shape[1] / params.num_pieces_horizontal
    for i in range(params.num_pieces_vertical):
        for j in range(params.num_pieces_horizontal):
            ii = np.random.choice(h - H, 1, replace=False)[0]
            jj = np.random.choice(w - W, 1, replace=False)[0]

            area_to_match = \
                params.image[int(ii * unit_height_rescale / H):int((ii + H) * unit_height_rescale / H),
                int(jj * unit_width_rescale / W):int((jj + W) * unit_width_rescale / W), :]
            img_mosaic[ii: ii + H, jj: jj + W, :] = get_best_match(params, area_to_match)
            print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
