MANUAL DE EJECUCIÓN

Los datasets y los elementos necesarios para ejecutar las aplicaciones estarán en la carpeta resources.
También se encuentra en el repositorio la aplicación para entrenar las redes neuronales (RNNCNNCRNN.py) y la aplicación con la interfaz para poder 
interpretar banderas (inter3.py). Si se quiere ejecutar estas aplicaciones tan solo es necesario tener en la misma ruta la carpeta resources, 
abrir el entorno de desarrollo pycharm, ya que es el entorno en el cual se ha realizado el proyecto, después se necesita instalar las librerías necesarias 
y ya solo es necesario ejecutar. 
Si la intención es reentrenar los modelos hay que eliminar de la carpeta resources los archivos .h5 si no, la aplicación simplemente te pedirá una imagen
y te dará la predicción. Puesto que el archivo saved_model_RNN.h5 era demasiado pesado para Github se tendría que reentrenar. 
Si la intención es crear otro ejecutable se deberá usar la línea comentada en la línea 15 del archivo inter3.py en la consola de comandos de pycharm.
Para utilizar el ejecutable solo es necesario descargarlo, darle doble click y esperar a que ejecute. Además de esto es necesario darle a permitir el archivo cuando el 
antivirus detecte este ejecutable como un virus ya que los antivirus detectan este tipo de ejecutables sin certificados como malware.
