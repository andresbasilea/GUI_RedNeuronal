
from tkinter import *
import mainReconocimientoNumeros

# Crear la GUI para poder dibujar un numero

ancho = 300
alto = 300

def pintar(event):
    x1, y1 = (event.x - 7), (event.y - 7)
    x2, y2 = (event.x + 7), (event.y + 7)
    w.create_oval(x1,y1,x2,y2, fill = 'black',outline = 'black')


master = Tk()
master.title("Reconocimiento por red neuronal de números")
anchoPantalla = master.winfo_screenwidth()
altoPantalla = master.winfo_screenheight()
x = (anchoPantalla/2) - (ancho/2)
y = (altoPantalla/2) - (alto/2)
master.geometry('%dx%d+%d+%d' % (ancho+200,alto+150,x,y))
w = Canvas(master,width = ancho, height = alto, background = 'white')
w.pack(expand = YES, fill = BOTH)
w.bind("<B1-Motion>",pintar)

z = Canvas(master, width = ancho - 100, height = alto - 100, background = 'turquoise')
z.pack(expand = YES, fill = BOTH,side = RIGHT)
id_texto = z.create_text(50, 50, fill='navy', font='Times 70 bold')

m = Label(master, text = "Dibuje un número presionando el mouse", font = 'Serif')
m.pack(side = TOP)

etiqueta = Label(master, text='Esperando número', font = 'Serif')
etiqueta.config(bg='white')
etiqueta.pack()


boton3 = Button(master, text = 'Ingresar nuevo número')
boton3.config(bg = 'white', font = 'Serif')
boton3.pack(side = BOTTOM)

boton = Button(master, text = 'Analizar')
boton.config(bg = 'white', font = 'Serif')
boton.pack(side = BOTTOM)



def analizar():
    global id_texto
    etiqueta.config(text = "Número recibido. Analizando... ", font = 'Serif')
    numero = mainReconocimientoNumeros.main()
    id_texto = z.create_text(100, 100, fill='navy', font='Times 70 bold', text=numero)

def ingresarNuevo():
    global id_texto
    w.delete('all')
    etiqueta.config(text = 'Esperando número',font = 'Serif')
    z.delete(id_texto)

boton.config(command = analizar)
boton3.config(command = ingresarNuevo)


mainloop()


