import sys
import cvlib
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load


kernel_cross = np.array([[  0, 255,  0],
                         [255, 255, 255],
                         [  0, 255,  0]], np.uint8)



def find_plate(input_filename, b = 1, dila = 1):
    """ Detecta la placa y regresa la ubicación y las dimensiones de la placa en la imagen. 

    Args:
        - input_filename (str): path de la imagen de la cual se quiere obtener la imagen 
        - b (int, optional): intensidad del blur que se le aplicará a la imagen. Defaults to 1.
        - dila (int, optional): intensidad de la dilatación que se aplicará a la imagen. Defaults to 1.

    Returns:
        - x (int): posición en x donde se encontró la placa  
        - y (int): posición en y donde se contró la placa
        - w (int): ancho de la placa
        - h (int): alto de la placa 

        Todo se devuelve en una tupla.
    """
    
    print("---DETECCIÓN DE PLACA---")
    # imagen a blanco y negro 
    img_gray = cv.imread(input_filename, cv.IMREAD_GRAYSCALE)
    #cvlib.imgview(img_gray, title = "Grayscale")

    # imagen a color
    img_color = cv.imread(input_filename,cv.IMREAD_COLOR) 
    img_color = cv.cvtColor(img_color,cv.COLOR_BGR2RGB)
    #cvlib.imgview(img_color, title= "Color")

    # Blur para imagenes 
    img_blur = img_gray.copy()
    if b > 0: 
        img_blur = cv.blur(img_gray,(b,b))
        t = "Blur " + str(b)
    

    # Binarizacion 
    imgbin = cv.adaptiveThreshold(img_blur, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,5)


    # Dilatacion
    dilateded = cv.dilate(imgbin, kernel_cross, iterations = dila)

    # Generador de contornos 
    mode = cv.RETR_TREE 
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE]
    contours, hierarchy = cv.findContours(dilateded, mode, method[1])

    # Impresion de todos los contornos
    color = (0,255,0) #(r,g,b)
    thickness = 2
    r1 = cv.cvtColor(imgbin.copy(),cv.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        r1 = cv.drawContours(r1, contours, i, color, thickness)


    # Buqueda del contorno más cuadrado
    indice = 0
    extent_mayor = 0.0

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        rect_area = w*h
        extent = float(area)/rect_area

        area_imagen = img_gray.shape[0] * img_gray.shape[1]
        area_de_placa = area / area_imagen

        
        if w >= h:
            if (extent > extent_mayor): #and (len(contours_internos) > 1):
                indice = i
                extent_mayor = extent
        else:
            if (extent > extent_mayor) and (area_de_placa >= 0.04): #and (len(contours_internos) > 1):
                indice = i
                extent_mayor = extent

    # Impresión del contorno más ajutado
    index =  indice
    color = (0,255,0) #(r,g,b)
    thickness = 2
    r3 = img_color.copy()
    r3 = cv.drawContours(r3, contours, index, color, thickness)



    # Area de la placa sobre área de la imagen  
    area = cv.contourArea(contours[index])
    area_imagen = img_gray.shape[0] * img_gray.shape[1]
    area_de_placa = area / area_imagen
    
    # Corte de imagen
    x,y,w,h = cv.boundingRect(contours[index])


    print("Listo.")
    return x, y, w, h



def find_numbers2(input_filename):
    """ Recibe el path de la placa obtiene la placa por medio de la función find_plate y obtiene los números de la placa con ayuda de una modelo 

    Args:
        input_filename (str): path de la imagen de entrada

    return: 
        resultado (str): string con los caracteres de la placa, si no se detecta nada regresa None. 
    """

    # PASOS: 
    # - Detección de placas (LISTO)
    # - Lectura de las imágenes (LISTO)
    # - Detección de fondo claro o oscuro (LISTO)
    # -- Si el fondo es claro determinar si no es rojo
    # - Binarización dependiendo de si es claro o oscuro  (LISTO)
    # - Determinación de tipo de placas: tipo 1 horizontal 1 línea, tipo 2 horizontal 2 líneas , tipo 3 vertical 2 líneas (LISTO)
    # - Obtención de contornos (LISTO)
    # - Filtro de contornos por sus proporciones (LISTO)
    # - Empaquetados de contornos en una función (LISTO)
    # - Determinar el orden de los números (LISTO)
    # - Recorte de los números y pasos a la función (LISTO)
    # - Detección de números en el modelo (LISTO)
    # - Impresión de la placa con los números (LISTO)
    # - Devolución de números (LISTO)

    # Deteccion de la placa
    x1, y1, w1, h1 = find_plate(input_filename, 1, 1)

    color = (0,255,0) 

    print("\n\n---DETECCIÓN DE NUMEROS---")
    # Lectura a color 
    img_color = cv.imread(input_filename,cv.IMREAD_COLOR) 
    img_color = cv.cvtColor(img_color,cv.COLOR_BGR2RGB)

    # Lectura en blaco y negro
    img_gray = cv.imread(input_filename, cv.IMREAD_GRAYSCALE)

    # Recorte - escala de grises
    img_re = img_gray[y1:y1+h1, x1:x1+w1]

    # Recorte - color
    img_re_color = img_color[y1:y1+h1, x1:x1+w1]


    # Binarización
    imgbin = img_re.copy()
    tresh = 91
    hold = 5

    imgbin = cv.adaptiveThreshold(img_re, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,tresh, hold)

    # Histograma de binarización
    histr_imgbin = cv.calcHist([imgbin],None,None,[256],[0,256])

    if histr_imgbin[0][0] < histr_imgbin[255][0]:
        imgbin = 255 - imgbin
    

    # Opening - erosion seguido de dilatation
    kernel_cross = np.array([[  0, 255,  0],
                         [255, 255, 255],
                         [  0, 255,  0]], np.uint8)
    imgbin = cv.morphologyEx(imgbin, cv.MORPH_OPEN, kernel_cross)
    
    # Obtención de contornos
    mode = cv.RETR_TREE 
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE]
    contours = cv.findContours(imgbin, mode, method[1])[0]



    # CALCULO DE PROPORCIÓN altura/ancho 
    proporcion = float(h1 / w1)
    # print("Proporción (Altura/Ancho):", float(proporcion)) 


    # Información que se guardará de los números
    tipo = "" # tipo de placa 
    lista_numeros = [] # lista de contornos
    lista_rec = [] # lista de imágenes recortadas y preporsedas para el modelo 
    Xs = [] # Ubicación en X
    Ys = [] # Ubicación en Y
    Ws = [] # Ancho
    Hs = [] # Alto
    nivel = [] # Indica si el caracter se encuentra en primera o segundo línea

    # Imagen para graficar los contornos
    r2 = cv.cvtColor(imgbin.copy(),cv.COLOR_GRAY2RGB) 
    

    # DETECCIÓN DE NÚMEROS POR TIPO DE PLACA 
    # Tipos: tipo 1 (horizontal 1 línea), tipo 2 (horizontal 2 líneas) y tipo 3  ()
    padding = 6
    if proporcion <= 0.36:
        print("TIPO 1: horizonal 1 línea")
        tipo = "tipo1"

        for cnt in contours: 
            x,y,w,h = cv.boundingRect(cnt)

            proporcion_altura = float(h/h1)
            # proporcion_ancho = float(w/h)

            if (h > w) and (proporcion_altura >= 0.60) and (proporcion_altura <= 0.85) and (x > 0) and ((x+w) < (x1+w1)) : # and (proporcion_ancho > 0.2): # Si la altura es mayor al ancho se guarda el contorno
                
                # recorte del rectangulo
                rectangulo = imgbin[y:y+h, x:x+w]
                rectangulo = np.pad(rectangulo, padding, 'constant', constant_values = 0)
                rectangulo = 255 - rectangulo # Línea que se cambia con el modelo binarizado
                rectangulo = cv.resize(rectangulo, (75, 100))
                
                r2 = cv.rectangle(r2,(x,y),(x+w,y+h),(0,255,0),1)


                # Guardado de información del digito
                lista_numeros.append(cnt)
                lista_rec.append(rectangulo.flatten()) # El recorte se guarda como flaten
                Xs.append(x)
                Ys.append(y)
                Ws.append(w)
                Hs.append(h)

        # Clasificación de los digitos por línea     
        if len(Ys) > 0:
            min_y = np.min(Ys)

            for i in Ys:
                if i < (min_y + 15): 
                    nivel.append(1)
                else: 
                    nivel.append(2)

    elif proporcion > 0.36 and proporcion < 1.0:
        print("TIPO 2: horizonal 2 líneas")

        tipo = "tipo2"
        
        for cnt in contours: 
            x,y,w,h = cv.boundingRect(cnt)

            proporcion_altura = float(h/h1)

            if (h > w) and (proporcion_altura >= 0.30) and (proporcion_altura <= 0.45) and (x > 0) and ((x+w) < (x1+w1)) : # and (proporcion_ancho > 0.2): # Si la altura es mayor al ancho se guarda el contorno
                

                # recorte del rectangulo
                rectangulo = imgbin[y:y+h, x:x+w]
                rectangulo = np.pad(rectangulo, padding, 'constant', constant_values = 0)
                rectangulo = 255 - rectangulo # Línea que se cambia con el modelo binarizado
                rectangulo = cv.resize(rectangulo, (75, 100))
                
                r2 = cv.rectangle(r2,(x,y),(x+w,y+h),(0,255,0),1)

                # Guardado de información del digito
                lista_numeros.append(cnt)
                lista_rec.append(rectangulo.flatten())
                Xs.append(x)
                Ys.append(y)
                Ws.append(w)
                Hs.append(h)

        # Clasificación de los digitos por línea 
        if len(Ys) > 0:
            min_y = np.min(Ys)

            for i in Ys:
                if i < (min_y + 15): 
                    nivel.append(1)
                else: 
                    nivel.append(2)

    else: 
        print("TIPO 3: vertival 2 líneas")

        tipo = "tipo3"

        for cnt in contours: 
            x,y,w,h = cv.boundingRect(cnt)

            proporcion_altura = float(h/h1)

            if (h > w) and (proporcion_altura >= 0.25) and (proporcion_altura <= 0.45) and (x > 0) and ((x+w) < (x1+w1)) : # and (proporcion_ancho > 0.2): # Si la altura es mayor al ancho se guarda el contorno

                
                # recorte del rectangulo
                rectangulo = imgbin[y:y+h, x:x+w]
                rectangulo = np.pad(rectangulo, padding, 'constant', constant_values = 0)
                rectangulo = 255 - rectangulo # Línea que se cambia con el modelo binarizado
                rectangulo = cv.resize(rectangulo, (75, 100))
                
                # Guardado de información del digito
                r2 = cv.rectangle(r2,(x,y),(x+w,y+h),(0,255,0),1)
                lista_numeros.append(cnt)
                lista_rec.append(rectangulo.flatten())
                Xs.append(x)
                Ys.append(y)
                Ws.append(w)
                Hs.append(h)

        
        # Clasificación de los digitos por línea 
        if len(Ys) > 0:
            min_y = np.min(Ys)

            for i in Ys:
                if i < (min_y + 15): 
                    nivel.append(1)
                else: 
                    nivel.append(2)

    
    # Dataframe con los digitos de la placa 
    numeros = pd.DataFrame({
        'X': Xs,
        'Y': Ys,
        'W': Ws,
        'H': Hs, 
        'Nivel': nivel, 
        'Contornos': lista_numeros, 
        'Rectangulos': lista_rec
    })




    # DETECCION DE DIGITOS
    if numeros.shape[0] > 0:  # Si no se encontraron caracteres no se hará la predicción. 
        lista = []
        lista2 = []

        if tipo == "tipo1":
            lista = numeros.sort_values(by=["X", "Y"], ascending=True).to_numpy()
            lista2 = numeros.sort_values(by=["X", "Y"], ascending=True)
        elif (tipo == "tipo2") or (tipo == "tipo3") :
            lista_ordenada = numeros.sort_values(by=["Nivel","X"], ascending=True).groupby('Nivel')
            
            lista = lista_ordenada.head(numeros.shape[0]).reset_index(drop=True).to_numpy()
            lista2 = lista_ordenada.head(numeros.shape[0]).reset_index(drop=True)


        # Impresión de orden de los digitos 
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2

        for i in range(len(lista)): 
            org = (lista[i][0]+4, lista[i][1])
            r2 = cv.putText(r2, str(i), org, font, 
                        fontScale, color, thickness, cv.LINE_AA)
    
        # PREDICCIÓN 
        modelo = load('modelos/modelo.joblib') 
        label = load('modelos/label_encoder.joblib')

        # Preparación del input
        numeros_input = lista2[["Rectangulos"]].to_numpy()[:,0]
        numeros_ouput = modelo.predict(numeros_input.tolist())
        numeros_ouput = label.inverse_transform(numeros_ouput)

        # Eliminación de las primeras 2 letras I 
        if len(numeros_ouput) > 2: 
            if (numeros_ouput[0] == "I") and (numeros_ouput[1] == "I"):
                
                numeros_ouput = numeros_ouput[2:]

                lista2 = lista2.iloc[2:]
                

        print("Lista de dígitos:", numeros_ouput)
        print("Digitos de la placa:", "".join(numeros_ouput))


        # IMPRESIÓN DE LA IMAGEN FINAL 
        thickness = 2
        color = (0,255,0) #(r,g,b)

        # PLACA 
        imagen_final = img_color.copy()
        imagen_final = cv.rectangle(imagen_final,(x1,y1),(x1+w1,y1+h1),color,thickness)

        # DIGITOS
        for fila in lista2.to_numpy():
            x, y, w, h = fila[0:4]

            x = x + x1
            y = y + y1
            
            imagen_final = cv.rectangle(imagen_final,(x,y),(x+w,y+h),color,thickness)

        # TEXTO
        resultado = "".join(numeros_ouput)
        org = (x1, y1-4)
        imagen_final = cv.putText(imagen_final, resultado, org, font, 
                        fontScale, color, thickness, cv.LINE_AA)

        cvlib.imgview(imagen_final)

        print("\n\n")

        return resultado
    
    else: 
        color = (0,255,0) 
        thickness = 2
        imagen_final = img_color.copy()
        imagen_final = cv.rectangle(imagen_final,(x1,y1),(x1+w1,y1+h1),color,thickness)
        cvlib.imgview(imagen_final)
        print("No se detectaron los dígitos")

        print("\n\n")
        return None
    



# EJECUCIÓN
args = sys.argv
bandera = "--p"
path =  "fprint3.pgm"


# BANDERA
try:
    bandera = args[1]
    if bandera == "--p":
        pass
    else:
        print(f"warning: {bandera} no es una argumento conocido del programa")
        # print("en su lugar use '--p' para indicar el path.")
        exit(0)

except:
    print("usage: debe incluir '--p' para indicar la dirección del archivo.")
    exit(0)


# PATH 
try:
    path = args[2]
except:
    print("warning: debe indicar el path del archico a detectar.")
    exit(0)


print("Filename:", path)


find_numbers2(path)

print("--- FIN DEL PROGRAMA--")
# print("Output filename:", output_filename)