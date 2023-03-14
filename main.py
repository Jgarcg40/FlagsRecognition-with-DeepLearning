
# # Reconocimiento de banderas usando Deep Learning con una CNN

# # Imports
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image # type: ignore
from keras.preprocessing.image import ImageDataGenerator

# # Semilla y directorios


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_dir = r'C:\Users\yoni8\Desktop\dataset_train'
validation_dir = r'C:\Users\yoni8\Desktop\dataset_validation'
test_dir = r'C:\Users\yoni8\Desktop\dataset_test'

# # Etiquetado de clases


train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

directories = sorted(glob.glob(os.path.join(train_dir, '*')))
labels = [os.path.basename(d) for d in directories]
print(labels)

train_labels = []
for d in directories:
    label = os.path.basename(d)
    files = glob.glob(os.path.join(d, '*'))
    train_labels += [labels.index(label)] * len(files)

#print(train_labels)

# # Cargar train_dataset y validation_dataset

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(124, 124),
    batch_size=20,
    class_mode='categorical',
    shuffle=True,
    seed=42)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(124, 124),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)

# # Parámetros entrenamiento


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(124, 124, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

# # Compilar modelo


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # Cargar modelo ya entrenado


# Ejecutar solo si se quiere usar un modelo ya entrenado
model = models.load_model(r'C:\Users\yoni8\Desktop\saved_model.h5')


model.summary()

# # Entrenar modelo

"""
early_stop = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
history = model.fit(
    train_generator,
    steps_per_epoch=150,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=150,
    callbacks=[early_stop])

# # Guardar modelo para no perderlo

"""

#model.save(r'C:\Users\yoni8\Desktop\saved_model.h5')

# # Evaluar modelo


test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(124, 124),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test accuracy:', test_acc)

# # Estadísticas entrenamiento

"""
val_accuracy = history.history['val_accuracy']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', linewidth=2, linestyle='solid', label='Accuracy')
plt.plot(epochs, val_accuracy, 'b', linewidth=2, linestyle='solid', label='Validation accuracy')
plt.title('Accuracy y val_accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', linewidth=2, linestyle='solid', label='Loss')
plt.plot(epochs, val_loss, 'b', linewidth=2, linestyle='solid', label='Validation loss')
plt.title('Loss y val_loss')
plt.legend()
plt.show()
"""
# # Predicción

img_path = r'C:\Users\yoni8\Desktop\españa.jpg'
img = image.load_img(img_path, target_size=(124, 124))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

predictions = model.predict(img_tensor)
predicted_label_idx = np.argmax(predictions)
predicted_label = labels[predicted_label_idx]
print('Predicted label:', predicted_label)

sorted_idxs = np.argsort(predictions[0])[::-1]

for idx in sorted_idxs:
    label = labels[idx]
    prob = predictions[0][idx]
    print('{}: {:.4%}'.format(label, prob))


