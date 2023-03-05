from PIL import Image
import os

# Función recursiva para convertir imágenes a JPG
def convertir_a_jpg(ruta):
    for ruta_actual, carpetas, archivos in os.walk(ruta):
        for archivo in archivos:
            if archivo.endswith(".png") or archivo.endswith(".jpeg") or archivo.endswith(".bmp"):
                ruta_completa = os.path.join(ruta_actual, archivo)
                imagen = Image.open(ruta_completa)
                # Si la imagen tiene canal alpha(tercer canal), convertirla a RGB
                if imagen.mode == 'RGBA':
                    imagen = imagen.convert('RGB')
                # Si la imagen está en modo "P", convertirla a RGB
                elif imagen.mode == 'P':
                    imagen = imagen.convert('RGB')
                imagen.save(os.path.splitext(ruta_completa)[0] + ".jpg")

        for carpeta in carpetas:
            convertir_a_jpg(os.path.join(ruta_actual, carpeta))

# carpeta donde estan las imagenes a combertir
ruta_principal = r"C:\Users\yoni8\Desktop\p11"

# llamada a la funcion
convertir_a_jpg(ruta_principal)
