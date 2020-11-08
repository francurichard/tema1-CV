from parameters import *
import numpy as np
import timeit
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def extract_hexagon(params: Parameters, img):
    return img[params.hexagon_mask[0], params.hexagon_mask[1]]


def replace_hexagon(params, IMG, img, start_x, start_y):
    start_x = int(start_x)
    start_y = int(start_y)
    IMG[np.add(params.hexagon_mask[0], start_x), np.add(params.hexagon_mask[1], start_y), :] = img
    return IMG


def get_best_match(params: Parameters, area_to_match, metric='euclidean'):
    if params.hexagon:
        mean_color = np.mean(area_to_match, axis=0)
    else:
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
    N, H, W, C = params.small_images_shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                area_to_match = \
                    params.image_resized[i * H:(i + 1) * H, j * W:(j + 1) * W, :]
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
    N, H, W, C = params.small_images_shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    for i in range(params.num_pieces_vertical):
        for j in range(params.num_pieces_horizontal):
            ii = np.random.choice(h - H, 1, replace=False)[0]
            jj = np.random.choice(w - W, 1, replace=False)[0]

            area_to_match = params.image_resized[ii:ii + H, jj:jj + W, :]
            img_mosaic[ii: ii + H, jj: jj + W, :] = get_best_match(params, area_to_match)
            print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images_shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    for i in range(params.num_pieces_vertical):
        for j in range(params.num_pieces_horizontal):
            # print(i)
            # print(j)
            area_to_match = params.image_resized[i * H:(i + 1) * H, int(j * 1.5 * W): int(j * 1.5 * W + W), :]
            area_to_match = extract_hexagon(params, area_to_match)
            img_mosaic = replace_hexagon(
                params,
                img_mosaic,
                get_best_match(params, area_to_match),
                i * H,
                j * 1.5 * W)
            if i != params.num_pieces_vertical - 1:
                area_to_match = params.image_resized[i * H + H // 2: (i + 1) * H + H // 2,
                                int(3 * W // 4 + j * 1.5 * W):int(3 * W // 4 + j * 1.5 * W + W), :]
                area_to_match = extract_hexagon(params, area_to_match)
                img_mosaic = replace_hexagon(
                    params,
                    img_mosaic,
                    get_best_match(params, area_to_match),
                    i * H + H // 2,
                    3 * W // 4 + j * 1.5 * W
                )
            print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))


    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))
    return img_mosaic
