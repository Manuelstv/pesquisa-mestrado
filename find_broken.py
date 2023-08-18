import os
from PIL import Image
from skimage.io import imread

# Specify the directory containing PNG files
directory = '/home/mstveras/struct3d-data'

# Iterate through files and detect broken PNGs
for filename in os.listdir(directory):
    if filename.lower().endswith('.png'):
        try:
            img = imread(f'/home/mstveras/struct3d-data/{filename}')
        except Exception as e:
            print(f"Broken PNG file: {filename}")
