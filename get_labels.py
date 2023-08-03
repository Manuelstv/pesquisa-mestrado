import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

dictionary = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "other structure",
    39: "other furniture",
    40: "other prop"
}

cores = np.array([
    #(0, 0, 0),
    (174, 199, 232),    # wall
    (152, 223, 138),    # floor
    (31, 119, 180),     # cabinet
    (255, 187, 120),    # bed
    (188, 189, 34),     # chair
    (140, 86, 75),      # sofa
    (255, 152, 150),    # table
    (214, 39, 40),      # door
    (197, 176, 213),    # window
    (148, 103, 189),    # bookshelf
    (196, 156, 148),    # picture
    (23, 190, 207),     # counter
    (178, 76, 76),
    (247, 182, 210),    # desk
    (66, 188, 102),
    (219, 219, 141),    # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),     # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),    # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),      # toilet
    (112, 128, 144),    # sink
    (96, 207, 209),
    (227, 119, 194),    # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),      # otherfurn
    (100, 85, 144)
])

root_dir = '/home/mstveras/pesquisa-mestrado/Structured3D'

df = pd.DataFrame()


for dirpath, dirnames, filenames in os.walk(root_dir):
    # Verifique se a pasta contém a subpasta 'full'
    if 'full' in dirnames:
        # Construa o caminho completo para a imagem semantic.png
        image_path = os.path.join(dirpath, 'full', 'semantic.png')
        
        # Verifique se a imagem semantic.png existe
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Erro ao ler a imagem: {image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            unique_colors = np.unique(img.reshape(-1, 3), axis=0)
            indices_cores = tuple(np.where(np.all(cores[:, None] == unique_colors, axis=2))[0])
            
            #indices = str(indices_cores.item()
            # Dividir a string pelo caractere '/'
            parts = image_path.split('/')
            # Encontrar a parte que contém "scene_*"
            scene_part = [part for part in parts if part.startswith("scene_")]
            print(parts)
            print(unique_colors)
            row_data = {
            'col1': parts[5],
            'col2': parts[7],
            'col3': indices_cores
            }
            df = df.append(row_data, ignore_index=True)

# Convert integer values in col2 to strings
df['col2'] = df['col2'].astype(str)
df['col1'] = df['col1'].astype(str)

# Combine the first two columns into one
df['combined'] = df['col1'] + '_' + df['col2']+'.png'

# Get all unique values from the third column
unique_values = set()
for row in df['col3']:
    unique_values.update(row)

# Create new columns with binary indicators
for value in range(0, 40):
    df[f'class{value}'] = df['col3'].apply(lambda x: 1 if value in x else 0)

# Drop unnecessary columns
df = df.drop(['col1', 'col2', 'col3'], axis=1)

T = 500

# Calculate the sum of each column
column_sums = df.iloc[:, 1:].sum()

# Create a boolean mask for columns whose sum is smaller than T
mask = column_sums < T

# Drop columns based on the mask
df = df.drop(df.columns[1:][mask], axis=1)

column_sums = df.iloc[:, 1:].sum()
# Create a boolean mask for columns whose sum is smaller than T
mask = column_sums > 600

# Drop columns based on the mask
df = df.drop(df.columns[1:][mask], axis=1)


df.to_csv('img_classes2.csv')

