# Sistema de Reconocimiento de Banderas con Redes Neuronales

## Descripción
Este proyecto implementa un sistema de reconocimiento de banderas de países en imágenes utilizando diferentes arquitecturas de redes neuronales profundas. Se desarrollaron y evaluaron tres tipos de redes neuronales (CNN, RNN y CRNN) para determinar cuál ofrece el mejor rendimiento en la tarea de clasificación de banderas de 193 países.

## Estructura del Proyecto
```
FlagsRecognition_WithDeepLearning-main/
├── aplicaciones de Python/        # Scripts para preparación de datos
│   ├── AugmentacionScript.py      # Aumentación de imágenes
│   └── ResizeImagesSript.py       # Redimensionado de imágenes
├── resources/                     # Datasets y modelos entrenados
│   ├── dataset_train/             # Conjunto de datos de entrenamiento
│   ├── dataset_validation/        # Conjunto de datos de validación
│   ├── dataset_test/              # Conjunto de datos de prueba
│   ├── saved_model_CNN.h5         # Modelo CNN entrenado (mejor rendimiento)
│   └── saved_model_CRNN.h5        # Modelo CRNN entrenado
├── RNNCNNCRNN.py                  # Script para entrenar y evaluar las tres redes
└── inter3.py                      # Aplicación de interfaz gráfica
```

## Características

### Preparación y Aumento de Datos
- Se crearon tres conjuntos de datos: entrenamiento, validación y prueba
- Cada país cuenta con 1000 imágenes aumentadas a partir de 5 imágenes originales
- Las imágenes se redimensionan a 124x124 píxeles
- Se aplicaron técnicas de aumentación: rotación, zoom, desplazamiento y volteo horizontal

### Arquitecturas Implementadas
1. **CNN (Red Neuronal Convolucional)**: Arquitectura tradicional con capas convolucionales y de agrupación, seguidas de capas densas. Demostró ser la más efectiva para este problema.
2. **RNN (Red Neuronal Recurrente)**: Combina capas convolucionales con una capa LSTM para capturar dependencias temporales.
3. **CRNN (Red Neuronal Convolucional Recurrente)**: Híbrido que aprovecha las características espaciales de las CNN y las temporales de las RNN.

## Requisitos e Instalación
- Python 3.x
- TensorFlow y Keras
- PIL/Pillow
- NumPy
- Tkinter
- Wikipedia API (`pip install wikipedia`)

## Uso

### Entrenamiento de Modelos
Para entrenar y evaluar los modelos, ejecute:
```
python RNNCNNCRNN.py
```
Este script permite seleccionar qué modelo entrenar (CNN, RNN o CRNN) y muestra estadísticas de rendimiento.

### Aplicación con Interfaz Gráfica
Para utilizar la aplicación de reconocimiento de banderas:
```
python inter3.py
```

Instrucciones de uso:
1. Haga clic en "Introducir Imagen" para seleccionar una imagen de bandera
2. Seleccione el modelo a utilizar (CNN recomendado por su mejor rendimiento)
3. La aplicación mostrará el país identificado y un breve texto informativo de Wikipedia

### Creación de Ejecutable
Para crear un ejecutable de la aplicación, descomentar y ejecutar la línea 15 de inter3.py:
```
pyinstaller --onefile --add-data "resources;resources" --icon="resources/mundo.ico" --name "Interpretador de banderas" -w inter3.py
```

## Resultados
Tras la evaluación comparativa, la arquitectura CNN demostró el mejor rendimiento en la clasificación de banderas, con mayor precisión y menor tiempo de procesamiento.

## Conclusiones
Este proyecto demuestra la efectividad de las redes neuronales convolucionales para la clasificación de imágenes de banderas. La CNN supera a las arquitecturas RNN y CRNN en este contexto específico, lo que sugiere que las características espaciales son más relevantes que las temporales para este problema de reconocimiento.
