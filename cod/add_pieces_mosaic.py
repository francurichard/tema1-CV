# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

from parameters import *
import numpy as np
import timeit


# extrag pixeli din interiorul hexagonului din imagine folosind masca creata in load_pieces
def extract_hexagon(params: Parameters, img):
    return img[params.hexagon_mask[0], params.hexagon_mask[1]]


# inlocuiesc hexagonul din imaginea
def replace_hexagon(params, IMG, img, start_x, start_y):
    start_x = int(start_x)
    start_y = int(start_y)
    IMG[np.add(params.hexagon_mask[0], start_x), np.add(params.hexagon_mask[1], start_y), :] = img
    return IMG


def get_best_match(params: Parameters, area_to_match, neighs):
    if params.hexagon:
        temp_neighs = []
        for neigh in neighs:
            temp_neighs.append(extract_hexagon(params, neigh))

        neighs = temp_neighs

    if params.hexagon:
        mean_color = np.mean(area_to_match, axis=0)
    else:
        mean_color = np.mean(area_to_match, axis=(0, 1))
    mean_color = mean_color.reshape(1, -1)
    min_, best = float('inf'), -1
    for i, mean_ in enumerate(params.small_images_mean_color):
        mean_ = mean_.reshape(1, -1)
        dist = np.linalg.norm(mean_color - mean_)
        if dist < min_:
            ok = True
            # am implementat doar pentru cazul cu poze dreptunghiulare
            if params.different_neighbors and params.hexagon is False:
                for neigh in neighs:
                    if (neigh == params.small_images[i]).all():
                        ok = False
            if ok:
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
                # pentru fiecare imagine iau veginul de sus si stanga si am grija sa fie diferiti
                neighs = []
                if params.different_neighbors:
                    if i != 0:
                        neighs.append(img_mosaic[(i - 1) * H:i * H, int(j * W): int((j + 1) * W), :])
                    if j != 0:
                        neighs.append(img_mosaic[i * H:(i + 1) * H, int((j - 1) * W): int(j * W), :])
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = get_best_match(params, area_to_match, neighs)
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
    for i in range(params.num_pieces_vertical * 2):
        for j in range(params.num_pieces_horizontal * 2):
            ii = np.random.choice(h - H, 1, replace=False)[0]
            jj = np.random.choice(w - W, 1, replace=False)[0]

            area_to_match = params.image_resized[ii:ii + H, jj:jj + W, :]
            img_mosaic[ii: ii + H, jj: jj + W, :] = get_best_match(params, area_to_match, [])
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
            area_to_match = params.image_resized[i * H:(i + 1) * H, int(j * 1.5 * W): int(j * 1.5 * W + W), :]
            area_to_match = extract_hexagon(params, area_to_match)
            img_mosaic = replace_hexagon(
                params,
                img_mosaic,
                get_best_match(params, area_to_match, []),
                i * H,
                j * 1.5 * W)
            if i != params.num_pieces_vertical - 1:
                area_to_match = params.image_resized[i * H + H // 2: (i + 1) * H + H // 2,
                                int(3 * W // 4 + j * 1.5 * W):int(3 * W // 4 + j * 1.5 * W + W), :]
                area_to_match = extract_hexagon(params, area_to_match)
                img_mosaic = replace_hexagon(
                    params,
                    img_mosaic,
                    get_best_match(params, area_to_match, []),
                    i * H + H // 2,
                    3 * W // 4 + j * 1.5 * W
                )
            print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    img_mosaic = img_mosaic[int(H // 2):int(h - H // 2), int(W // 2):int(w - W), :]
    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))
    return img_mosaic
