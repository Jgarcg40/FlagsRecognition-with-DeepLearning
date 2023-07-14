MANUAL DE EJECUCIÓN

Los datasets y los elementos necesarios para ejecutar las aplicaciones estarán en la carpeta resources, además en la carpeta aplicaciones python se encuentra los scripts usados para lograr los datasets.
También se encuentra en el repositorio la aplicación para entrenar las redes neuronales (RNNCNNCRNN.py) y la aplicación con la interfaz para poder interpretar banderas (inter3.py). Si se quiere ejecutar estas aplicaciones tan solo es necesario tener en la misma ruta la carpeta resources, abrir el entorno de desarrollo pycharm, ya que es el entorno en el cual se ha realizado el proyecto, después se necesita instalar las librerías necesarias y ya solo es necesario ejecutar.

Se usaron 3 redes neuronales distintas para probar distintas eficacias e investigar, pero la que mejor funciona es la CNN. Los archivos .h5 de resources son las redes ya entrenadas para el uso de la interfaz,pero es necesario entrenar la RNN ya que era demasiado grande para guardar en github asi que no tengo el entrenamiento.

Si la intención es crear un ejecutable con toda la aplicación se deberá usar la línea comentada en la línea 15 del archivo inter3.py en la consola de comandos de pycharm.

