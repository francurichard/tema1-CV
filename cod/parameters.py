import cv2 as cv


# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
class Parameters:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv.imread(image_path)
        if (self.image[:, :, 0] == self.image[:, :, 1]).all() and (self.image[:, :, 1] == self.image[:, :, 2]).all():
            self.grayscale = True
        else:
            self.grayscale = False
        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)

        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None
        self.small_images_mean_color = None
        self.hexagon_mask = None
        self.small_images_shape = None
        self.different_neighbors = None
