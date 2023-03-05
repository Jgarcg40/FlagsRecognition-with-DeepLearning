import os

# Función recursiva para eliminar archivos que no son JPG
def eliminar_archivos_no_jpg_ni_png(ruta):
    for ruta_actual, carpetas, archivos in os.walk(ruta):
        for archivo in archivos:
            ruta_completa = os.path.join(ruta_actual, archivo)
            if not archivo.endswith(".jpg"):
                os.remove(ruta_completa)
                print(f"Archivo eliminado: {ruta_completa}")

        for carpeta in carpetas:
            eliminar_archivos_no_jpg_ni_png(os.path.join(ruta_actual, carpeta))

# Carpeta donde estan las imagenes
ruta_principal = r"C:\Users\yoni8\Desktop\p11"

# llamada a la funcion
eliminar_archivos_no_jpg_ni_png(ruta_principal)
