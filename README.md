# Sistema de Reconocimiento de Banderas con Redes Neuronales (2022)

## Descripción
Este proyecto de investigación implementa un sistema de reconocimiento de banderas de países utilizando diferentes arquitecturas de redes neuronales profundas. El objetivo principal fue comparar tres tipos de redes neuronales (CNN, RNN y CRNN) para determinar cuál ofrece el mejor rendimiento en la tarea de clasificación de banderas de 193 países. Los resultados experimentales demostraron que la CNN es la arquitectura más eficaz para este tipo de reconocimiento de imágenes, aunque la CRNN también mostró un rendimiento aceptable.

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
└── inter3.py                      # Aplicación de interfaz gráfica para uso final
```

## Características

### Preparación y Aumento de Datos
- Se crearon tres conjuntos de datos: entrenamiento, validación y prueba
- Cada país cuenta con 1000 imágenes aumentadas a partir de 5 imágenes originales
- Las imágenes se redimensionan a 124x124 píxeles
- Se aplicaron técnicas de aumentación: rotación, zoom, desplazamiento y volteo horizontal

### Arquitecturas Implementadas y Resultados
1. **CNN (Red Neuronal Convolucional)**: Arquitectura tradicional con capas convolucionales y de agrupación, seguidas de capas densas. **Demostró ser la más efectiva para este problema** con la mayor precisión en el reconocimiento de banderas.
2. **RNN (Red Neuronal Recurrente)**: Aunque las RNN están diseñadas principalmente para datos secuenciales (texto, series temporales), se incluyó en el estudio para explorar su comportamiento con datos de imagen. Como era de esperar, su rendimiento fue inferior a las otras arquitecturas para este caso específico.
3. **CRNN (Red Neuronal Convolucional Recurrente)**: Arquitectura híbrida que aprovecha las características espaciales de las CNN y las temporales de las RNN. Mostró un rendimiento cercano al de la CNN, pero sin superarla.

## Requisitos e Instalación
- Python 3.x
- TensorFlow y Keras
- PIL/Pillow
- NumPy
- Tkinter
- Wikipedia API (`pip install wikipedia`)

## Uso

### Aplicación con Interfaz Gráfica (Recomendado para usuarios finales)
Si solo desea utilizar el sistema de reconocimiento de banderas sin entrenar modelos, ejecute:
```
python inter3.py
```

Esta aplicación utiliza los modelos ya entrenados incluidos en la carpeta resources, por lo que no es necesario realizar el entrenamiento.

Instrucciones de uso:
1. Haga clic en "Introducir Imagen" para seleccionar una imagen de bandera
2. Seleccione el modelo a utilizar (CNN recomendado por su mejor rendimiento)
3. La aplicación mostrará el país identificado y un breve texto informativo de Wikipedia

### Entrenamiento de Modelos (Para investigadores)
Si desea repetir el experimento o entrenar los modelos con nuevos datos:
```
python RNNCNNCRNN.py
```
Este script permite seleccionar qué modelo entrenar (CNN, RNN o CRNN) y muestra estadísticas de rendimiento comparativo.

### Creación de Ejecutable
Para crear un ejecutable de la aplicación, descomentar y ejecutar la línea 15 de inter3.py:
```
pyinstaller --onefile --add-data "resources;resources" --icon="resources/mundo.ico" --name "Interpretador de banderas" -w inter3.py
```

## Resultados Experimentales
Tras la evaluación comparativa exhaustiva, la arquitectura CNN demostró el mejor rendimiento en la clasificación de banderas, con mayor precisión y menor tiempo de procesamiento. Esto confirma la hipótesis inicial de que las CNNs son más adecuadas para tareas de reconocimiento de imágenes estáticas como las banderas, donde los patrones espaciales son más relevantes que las relaciones temporales o secuenciales.

## Conclusiones
Este proyecto de investigación demuestra la efectividad de las redes neuronales convolucionales para la clasificación de imágenes de banderas. Aunque se exploraron arquitecturas alternativas como RNN y CRNN, la CNN supera a estas en el contexto específico del reconocimiento de banderas, lo que sugiere que para este tipo de tareas de clasificación de imágenes sin componente temporal, las CNN siguen siendo la opción más efectiva y eficiente.
