import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Ruta de la carpeta con las imagenes(se puede obtener con un for y una libreria de python pero yo las tenia ya guardadas asi que copie y pegue el nombre

data_dir = r"C:\Users\yoni8\Desktop\dataset_test"

# tamaño de las imágenes después de la transformación
img_size = 124

# parametros augmentacion
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


nombres_carpetas = os.listdir(data_dir)

for n in nombres_carpetas:
    # Las creamos
    ruta_archivo = os.path.join(data_dir, n)
    image_generator = datagen.flow_from_directory(
        ruta_archivo,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical')

    # Generamos las imagenes,1000 aprox

    count = 0
    while count < 1000:
        batch = next(image_generator)
        images, labels = batch
        for j in range(len(images)):
            img = images[j]
            label = labels[j]
            img_path = os.path.join(ruta_archivo, f'augmented_{count}.jpg')
            tf.keras.preprocessing.image.save_img(img_path, img)
            count += 1
    print(f"Aumento de datos aplicado a {count} imágenes en {ruta_archivo}.")
