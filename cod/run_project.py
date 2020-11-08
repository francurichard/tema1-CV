"""
    PROIECT MOZAIC
"""

# Parametrii algoritmului sunt definiti in clasa Parameters.

from parameters import *
from build_mosaic import *
import os

# os.chdir("cod/")

# numele imaginii care va fi transformata in mozaic
image_path = './../data/imaginiTest/ferrari.jpeg'
params = Parameters(image_path)

# directorul cu imagini folosite pentru realizarea mozaicului
params.small_images_dir = './../data/colectie/'
# tipul imaginilor din director
params.image_type = 'png'
# numarul de piese ale mozaicului pe orizontala
# pe verticala vor fi calcultate dinamic a.i sa se pastreze raportul
params.num_pieces_horizontal = 100
# afiseaza piesele de mozaic dupa citirea lor
params.show_small_images = False
# modul de aranjarea a pieselor mozaicului
# optiuni: 'aleator', 'caroiaj'
params.layout = 'aleator'
# criteriul dupa care se realizeaza mozaicul
# optiuni: 'aleator', 'distantaCuloareMedie'
params.criterion = 'distantaCuloareMedie'
# daca params.layout == 'caroiaj', sa se foloseasca piese hexagonale
params.hexagon = False

load_pieces(params)
img_mosaic = build_mosaic(params)
cv.imwrite('mozaic_aleator.png', img_mosaic)
