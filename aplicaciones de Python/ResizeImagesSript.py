import os
from PIL import Image

# ruta de la carpeta con magenes
data_folder = 'resources\dataset_train'
#funcion recursiva para que busque en carpetas dentro de otras carpetas
def resize_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                with Image.open(file_path) as img:
                    img = img.convert('RGB') #convierto a RGB en caso de que no lo sea
                    img_resized = img.resize((124, 124))#nuevo tamaño
                    img_resized.save(file_path)

resize_images_in_directory(data_folder)
