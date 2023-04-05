# # Imports
import contextlib
import glob
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
from keras import models
from tensorflow.keras.preprocessing import image  # type: ignore
import wikipedia
import re
import tensorflow as tf
#pyinstaller --onefile --add-data "resources;resources" --icon="resources/mundo.ico" --name "Interpretador de banderas" -w inter3.py


# se necesita hacer pip install wikipedia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#ventana principal

root = tk.Tk()
root.geometry('800x600')
root.title("Banderas")
root.configure(bg='#FFFFFF')
icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'mundo.ico'))

# Define el icono de la ventana
root.iconbitmap(default=icon_path)#cambia el icono de la barra de herramientas

# Cargar imagen de fondo
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'banderas2.png'))
img = Image.open(image_path)
img = img.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
bg = ImageTk.PhotoImage(img)

# Cargar imagen de fondo de los botnes
cnn_image = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'cielo.jpg'))

cnn_im = Image.open(cnn_image)
cnn_im = cnn_im.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
cnn_bg = ImageTk.PhotoImage(cnn_im)

# Crear canvas
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()


# Mostrar imagen de fondo ventana principal
canvas.create_image(0, 0, anchor=tk.NW, image=bg)




dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'dataset_test'))

# # Etiquetado de clases

directories = sorted(glob.glob(os.path.join(dir, '*')))
labels = [os.path.basename(d) for d in directories] #mostramos los nombres de la carpeta train_dir por orden

train_labels = []
for d in directories:
    label = os.path.basename(d)   #obtenemos los nombres de la carpeta train_dir
    files = glob.glob(os.path.join(d, '*'))
    train_labels += [labels.index(label)] * len(files) #etiquetamos cada foto de cada clase y metemos las etiquetas en train_labels


#final codigo entrenamiento
# Función para abrir y mostrar imagen seleccionada
img_path = ""
def open_image():
    global img_path
    img_tensor = None

    try:
        img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])#abre imágenes de este tipo solo
        if img_path:  # Verificar si se seleccionó una imagen
            img = Image.open(img_path).resize((124,124))
            img_tensor = np.array(img, dtype=np.float32)
            img_tensor /= 255.
            messagebox.showinfo(title="Imagen cargada", message="Has cargado la imagen correctamente")
        else:
            messagebox.showwarning(title="Advertencia", message="No se ha seleccionado ninguna imagen.")
    except Exception:
        messagebox.showerror(title="Error", message="error al cargar la imagen")

    return img_tensor



def get_summary(country): #funcion para texto de wikipedia
    wikipedia.set_lang("es")
    page = wikipedia.page(country)
    raw_summary = page.summary
    summary = re.sub(r'\[\d+\]', '', raw_summary)#eliminar simbolos
    summary_lines = summary.split('\n')[:1]#resumen
    summary = '\n'.join(summary_lines)
    return summary


# Función para ejecutar código del botón CNN
def cnn_function():
    dir2 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'saved_model_CNN.h5'))
    global img_path
    if not img_path:
        messagebox.showerror(title="Error", message="Debes introducir una imagen antes")
        return
    CNN_model = models.load_model(dir2)
    img = image.load_img(img_path, target_size=(124, 124))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    null_out = open(os.devnull, 'w')
    with contextlib.redirect_stdout(null_out):
        predictions = CNN_model.predict(img_tensor)
        predicted_label_idx = np.argmax(predictions)
        predicted_label = labels[predicted_label_idx]

    null_out.close()


    summary = get_summary(predicted_label) #wikipedia

    # Ventana
    cnn_window = tk.Toplevel(root)
    cnn_window.title("Resultado CNN")
    cnn_window.geometry('800x600')
    cnn_window.configure(bg='#99CCFF')


    canvas_crnn = tk.Canvas(cnn_window, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
    canvas_crnn.place(x=0, y=0)


    canvas_crnn.create_image(0, 0, anchor=tk.NW, image=cnn_bg)


    img = Image.open(img_path)
    img = img.resize((500, 500), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(cnn_window, image=img, bg='#99CCFF')
    panel.image = img
    panel.pack(side="top", pady=10)

    # predicción
    label = tk.Label(cnn_window, text=f"Bandera de {predicted_label}", font=("Arial", 20), bg='#FFFFFF')
    label.pack(side="top", pady=10)

    # wikipedia texto
    summary_label = tk.Label(cnn_window, text=summary, font=("Arial", 12), bg='#FFFFFF', wraplength=700,
                             justify=tk.LEFT)
    summary_label.pack(side="top", pady=10)


# Función para ejecutar código del botón RNN
def rnn_function():
    dir3 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'saved_model_RNN.h5'))

    global img_path
    if not img_path:
        messagebox.showerror(title="Error", message="Debes introducir una imagen antes")
        return
    RNN_model = models.load_model(dir3)
    img = image.load_img(img_path, target_size=(124, 124))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    null_out = open(os.devnull, 'w')
    with contextlib.redirect_stdout(null_out):
        predictions = RNN_model.predict(img_tensor)
        predicted_label_idx = np.argmax(predictions)
        predicted_label = labels[predicted_label_idx]

    null_out.close()
    summary = get_summary(predicted_label)
    rnn_window = tk.Toplevel(root)
    rnn_window.title("Resultado RNN")
    rnn_window.geometry('800x600')
    rnn_window.configure(bg='#99CCFF')
    canvas_crnn = tk.Canvas(rnn_window, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
    canvas_crnn.place(x=0, y=0)
    canvas_crnn.create_image(0, 0, anchor=tk.NW, image=cnn_bg)
    img = Image.open(img_path)
    img = img.resize((500, 500), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(rnn_window, image=img, bg='#99CCFF')
    panel.image = img
    panel.pack(side="top", pady=10)
    label = tk.Label(rnn_window, text=f"Bandera de {predicted_label}", font=("Arial", 20), bg='#FFFFFF')
    label.pack(side="top", pady=10)
    summary_label = tk.Label(rnn_window, text=summary, font=("Arial", 12), bg='#FFFFFF', wraplength=700,justify=tk.LEFT)
    summary_label.pack(side="top", pady=10)

# Función para ejecutar código del botón CRNN

def crnn_function():
    dir4 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources', 'saved_model_CRNN.h5'))
    global img_path
    if not img_path:
        messagebox.showerror(title="Error", message="Debes introducir una imagen antes")
        return

    CRNN_model = models.load_model(dir4)
    img = image.load_img(img_path, target_size=(124, 124))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    null_out = open(os.devnull, 'w')
    with contextlib.redirect_stdout(null_out):
        predictions = CRNN_model.predict(img_tensor)
        predicted_label_idx = np.argmax(predictions)
        predicted_label = labels[predicted_label_idx]

    null_out.close()
    summary = get_summary(predicted_label)
    crnn_window = tk.Toplevel(root)
    crnn_window.title("Resultado CRNN")
    crnn_window.geometry('800x600')
    crnn_window.configure(bg='#99CCFF')
    canvas_crnn = tk.Canvas(crnn_window, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
    canvas_crnn.place(x=0, y=0)
    canvas_crnn.create_image(0, 0, anchor=tk.NW, image=cnn_bg)
    img = Image.open(img_path)
    img = img.resize((500, 500), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(crnn_window, image=img, bg='#99CCFF')
    panel.image = img
    panel.pack(side="top", pady=10)
    label = tk.Label(crnn_window, text=f"Bandera de {predicted_label}", font=("Arial", 20), bg='#FFFFFF')
    label.pack(side="top", pady=10)
    summary_label = tk.Label(crnn_window, text=summary, font=("Arial", 12), bg='#FFFFFF', wraplength=700, justify=tk.LEFT)
    summary_label.pack(side="top", pady=10)



# boton para meter la imagen
btn_open = tk.Button(root, text="Introducir imagen", font=("Arial", 16), bg="#009933", fg="black", command=open_image)
btn_open.place(relx=0, rely=0.75, relwidth=1, relheight=0.15)

# botones redes neuronales
btn_cnn = tk.Button(root, text="CNN", font=("Arial", 16), bg="#66CCFF", fg="black",command=cnn_function)
btn_cnn.place(relx=0, rely=0.9, relwidth=0.333, relheight=0.1)

btn_rnn = tk.Button(root, text="RNN", font=("Arial", 16), bg="#E63D3D", fg="black",command=rnn_function)
btn_rnn.place(relx=0.333, rely=0.9, relwidth=0.333, relheight=0.1)

btn_crnn = tk.Button(root, text="CRNN", font=("Arial", 16), bg="#F7FF00", fg="black",command=crnn_function)
btn_crnn.place(relx=0.666, rely=0.9, relwidth=0.334, relheight=0.1)

root.mainloop()
