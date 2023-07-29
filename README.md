Aplicación para reconocer banderas en imágenes como proyecto de TFG de la carrera de ingeniería informática, probando 3 redes neuronales distintas para determinar cuál es la mejor y observar los distintos rendimientos de cada una. Se hicieron 3 datasets, uno de entrenamiento, otro se validación y uno de test, usándose en cada uno excepto en el de test, 1000 imágenes aumentadas a partir de 5 en cada país( total 193 países).


MANUAL DE EJECUCIÓN

Los datasets y los elementos necesarios para ejecutar las aplicaciones estarán en la carpeta resources. Además en la carpeta aplicaciones python se encuentra los scripts usados para lograr los datasets.
También se encuentra en el repositorio la aplicación para entrenar las redes neuronales (RNNCNNCRNN.py) y la aplicación con la interfaz para poder interpretar banderas (inter3.py). Si se quieren ejecutar estas aplicaciones tan solo es necesario tener en la misma ruta la carpeta resources, abrir el entorno de desarrollo pycharm, ya que es el entorno en el cual se ha realizado el proyecto, después se necesita instalar las librerías necesarias y ya sólo es necesario ejecutar.

Se usaron 3 redes neuronales distintas para probar distintas eficacias e investigar, pero la que mejor funciona es la CNN. Los archivos .h5 de resources son las redes ya entrenadas para el uso de la interfaz.

Si la intención es crear un ejecutable con toda la aplicación se deberá usar la línea comentada en la línea 15 del archivo inter3.py en la consola de comandos de pycharm.

Una vez realizados estos pasos para utilizar correctamente la aplicación con interfaz, sólo es necesario introducir una imagen con una bandera pulsando introducir imagen y seleccionandola de tu disco, y posteriormente pulsar el boton CNN u otra red neuronal si se prefiere. Después de eso mostrará el resultado de la predicción y un breve texto de Wikipedia hablando del país.
