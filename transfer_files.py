root_dir = '/home/mstveras/Structured3D'
#df = pd.DataFrame(columns=['col1', 'col2', 'col3'])

import shutil
import os

destination_folder = '/home/mstveras/struct3d-data'
new_filename = 'rgb_rawlight.png'

for dirpath, dirnames, filenames in os.walk(root_dir):
    #print(dirpath)
    # Verifique se a pasta contém a subpasta 'full'
    if 'full' in dirnames:
        # Construa o caminho completo para a imagem semantic.png
        image_path = os.path.join(dirpath, 'full', 'rgb_rawlight.png')
        
        # Verifique se a imagem semantic.png existe
        if os.path.exists(image_path):
            parts = image_path.split('/')
            # Encontrar a parte que contém "scene_*"
            scene_part = [part for part in parts if part.startswith("scene_")]
            
            shutil.copy2(image_path,f'{destination_folder}/{parts[4]}_{parts[6]}.png') 
