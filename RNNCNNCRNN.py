# # Imports
import glob
import os
import random
from tkinter import filedialog
from tkinter import *

import numpy as np
import tensorflow as tf
from keras import callbacks
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image  # type: ignore

# # Semilla y directorios


seed_value = 42   #semilla para que no varien los resultados en cada ejecucion
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True' #para evitar mensajes  de error

train_dir = 'resources\dataset_train'
validation_dir = 'resources\dataset_validation'  #directorios con los datasets
test_dir = 'resources\dataset_test'

# # Etiquetado de clases

train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

directories = sorted(glob.glob(os.path.join(train_dir, '*')))
labels = [os.path.basename(d) for d in directories] #mostramos los nombres de la carpeta train_dir por orden
print(labels)

train_labels = []
for d in directories:
    label = os.path.basename(d)   #obtenemos los nombres de la carpeta train_dir
    files = glob.glob(os.path.join(d, '*'))
    train_labels += [labels.index(label)] * len(files) #etiquetamos cada foto de cada clase y metemos las etiquetas en train_labels

#print(train_labels)

# # Cargar train_dataset y validation_dataset

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(124, 124),
    batch_size=20, #tamaño lote
    class_mode='categorical', #porque hay muchas clases
    shuffle=True, #mezclar aleatoriamente los datos
    seed=42)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(124, 124),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)

# # Parámetros entrenamiento

# Definir parámetros de la CNN
CNN_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(124, 124, 3)), #capa convolucional
    layers.MaxPooling2D((2, 2)), #reduce dimension espacial
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), #capa aplanamiento
    layers.Dense(512, activation='relu'), #capa densa con 512 neuronas
    layers.Dense(len(labels), activation='softmax') #producimos las probavilidades
])

# Definir parámetros de la RNN

RNN_model = models.Sequential([
    layers.Reshape((1, 124, 124, 3), input_shape=(124, 124, 3)), #remodelar a 4 dimensiones
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')), #usamos secuencias temporales
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(256),  #256 unidades LSTM
    layers.Dense(len(labels), activation='softmax')
])

#definir parámetros de la CRNN

CRNN_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(124, 124, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(256, return_sequences=True),
    layers.GlobalMaxPooling1D(),  # reduce la dimensión de la salida de la capa LSTM seleccionando el valor máximo de cada canal de características
    layers.Dense(len(labels), activation='softmax')
])


# Compilar modelos
CNN_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

RNN_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

CRNN_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

model_trained_CNN = False
model_trained_RNN = False
model_trained_CRNN = False
# Cargar modelo CNN ya entrenado
try:
    CNN_model = models.load_model('resources\saved_model_CNN.h5')
    print("Modelo CNN cargado")
    history_CNN = CNN_model.history #guardar estadisticas
    model_trained_CNN = True

except:
    pass

# Cargar modelo RNN ya entrenado
try:
    RNN_model = models.load_model('resources\saved_model_RNN.h5')
    print("Modelo RNN cargado")
    history_RNN =RNN_model.history
    model_trained_RNN = True

except:
    pass

# Cargar modelo CRNN ya entrenado
try:
    CRNN_model = models.load_model('resources\saved_model_CRNN.h5')
    print("Modelo CRNN cargado")
    history_RNN =CRNN_model.history
    model_trained_CRNN = True

except:
    pass

# Entrenar/Evaluar modelos
while True:

    choice = input("Ingrese 1 para CNN , 2 para RNN y 3 para la CRNN: ")
    Error = False

    if choice == '1':
      print("Has elegido ejecutar el modelo CNN")

      if not model_trained_CNN:

        # Entrenar modelo CNN
        print("Entrenando CNN")
        early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True) #con esto para cuando el modelo no mejora
        history_CNN = CNN_model.fit(train_generator,
                                steps_per_epoch=150,  #pasos entrenamiento
                                epochs=200,   #iteraciones
                                validation_data=validation_generator,
                                validation_steps=150,
                                callbacks=[early_stop])
        CNN_model.save('resources\saved_model_CNN.h5', include_optimizer=True) #guarda entrenamiento
        model_trained_CNN = True
        # Evaluar modelo CNN
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(test_dir, target_size=(124, 124), batch_size=20,
                                                          class_mode='categorical')
        test_loss, test_acc = CNN_model.evaluate(test_generator, steps=50)
        print('test accuracy:', test_acc)

        # # Estadísticas

        val_accuracy = history_CNN.history['val_accuracy']
        accuracy = history_CNN.history['accuracy']
        val_loss = history_CNN.history['val_loss']
        loss = history_CNN.history['loss']
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

      if model_trained_CNN:

        # Predicción
        root = Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename()
        img = image.load_img(img_path, target_size=(124, 124))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        predictions = CNN_model.predict(img_tensor)
        predicted_label_idx = np.argmax(predictions)
        predicted_label = labels[predicted_label_idx] #muestra la prediccion con su etiqueta
        print('Pais resultado:', predicted_label)

        sorted_idxs = np.argsort(predictions[0])[::-1]
        for idx in sorted_idxs:
            label = labels[idx]
            prob = predictions[0][idx]
            print('{}: {:.4%}'.format(label, prob)) #toda esta parte con el for para ver las provabilidades resto paises

      Error = True

    if choice == '2':
        print("Has elegido ejecutar el modelo RNN")
        if not model_trained_RNN:
            # Entrenar modelo RNN
            print("Entrenando RNN")
            early_stop2 = callbacks.EarlyStopping(monitor='val_accuracy', patience=35, restore_best_weights=True)
            history_RNN = RNN_model.fit(train_generator,
                                        steps_per_epoch=150,
                                        epochs=200,
                                        validation_data=validation_generator,
                                        validation_steps=150,
                                        callbacks=[early_stop2])
            RNN_model.save('resources\saved_model_RNN.h5',include_optimizer=True)
            model_trained_RNN = True

            # Evaluar modelo CNN
            test_datagen2 = ImageDataGenerator(rescale=1. / 255)
            test_generator = test_datagen2.flow_from_directory(test_dir, target_size=(124, 124), batch_size=20,
                                                               class_mode='categorical')
            test_loss, test_acc = RNN_model.evaluate(test_generator, steps=50)
            print('test accuracy:', test_acc)

            # # Estadísticas entrenamiento
            val_accuracy = history_RNN.history['val_accuracy']
            accuracy = history_RNN.history['accuracy']
            val_loss = history_RNN.history['val_loss']
            loss = history_RNN.history['loss']
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

        if model_trained_RNN:
            # Predicción

            root = Tk()
            root.withdraw()
            img_path = filedialog.askopenfilename()
            img = image.load_img(img_path, target_size=(124, 124))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.

            predictions = RNN_model.predict(img_tensor)
            predicted_label_idx = np.argmax(predictions)
            predicted_label = labels[predicted_label_idx]
            print('Pais resultado:', predicted_label)

            sorted_idxs = np.argsort(predictions[0])[::-1]
            for idx in sorted_idxs:
                label = labels[idx]
                prob = predictions[0][idx]
                print('{}: {:.4%}'.format(label, prob))

        Error = True

    if choice == '3':
        print("Has elegido ejecutar el modelo CRNN")
        if not model_trained_CRNN:
            # Entrenar modelo CRNN
            print("Entrenando CRNN")
            early_stop3 = callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
            history_CRNN = CRNN_model.fit(train_generator,
                                        steps_per_epoch=150,
                                        epochs=200,
                                        validation_data=validation_generator,
                                        validation_steps=150,
                                        callbacks=[early_stop3])
            CRNN_model.save('resources\saved_model_CRNN.h5',include_optimizer=True)
            model_trained_CRNN = True

            # Evaluar modelo CNN
            test_datagen3 = ImageDataGenerator(rescale=1. / 255)
            test_generator = test_datagen3.flow_from_directory(test_dir, target_size=(124, 124), batch_size=20,
                                                               class_mode='categorical')
            test_loss, test_acc = CRNN_model.evaluate(test_generator, steps=50)
            print('test accuracy:', test_acc)

            # # Estadísticas entrenamiento
            val_accuracy = history_CRNN.history['val_accuracy']
            accuracy = history_CRNN.history['accuracy']
            val_loss = history_CRNN.history['val_loss']
            loss = history_CRNN.history['loss']
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

        if model_trained_CRNN:
            # Predicción

            root = Tk()
            root.withdraw()
            img_path = filedialog.askopenfilename()
            img = image.load_img(img_path, target_size=(124, 124))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.

            predictions = CRNN_model.predict(img_tensor)
            predicted_label_idx = np.argmax(predictions)
            predicted_label = labels[predicted_label_idx]
            print('Pais resultado:', predicted_label)

            sorted_idxs = np.argsort(predictions[0])[::-1]
            for idx in sorted_idxs:
                label = labels[idx]
                prob = predictions[0][idx]
                print('{}: {:.4%}'.format(label, prob))

        Error = True

    if not Error:

        print("Error, ingrese 1 para CNN, 2 para RNN o 3 para CRNN.")

    else:

        print("¿Quieres salir de la aplicación? Escribe 'no' y si quieres permanecer toca una letra o numero cualquiera:")
        respuesta = input()
        if respuesta.lower() == 'no':
            break
#end