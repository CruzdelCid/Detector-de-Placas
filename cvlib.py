import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use('dark_background')


def imgview(img, size = 8, title = None, filename = None, axis = False):
    """Imprime la imagen a partir de un numpy array.

    Args:
        img (numpy array): contiene la información de la imagen en un formato uint8
        size (int): define el tamaño de impresión de la gráfica
        title (str): título de la imagen
        filename (str): nombre del archivo donde se almacenará la imagen
        axis (bool): bandera para indicar si imprimen los ejes
    """

    f, c = img.shape[0:2]
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    
    # verifica si la imagen está a color o escala de grises
    if len(img.shape) == 3: 
        im = ax.imshow(img,extent=None)
    else: 
        im = ax.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)

    # coloca un título
    if title:
        ax.set_title(title,fontsize=14)


    # agrega escala
    if not axis:
        plt.axis('off')
    else:
        ax.grid(c='b')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('b')
        ax.yaxis.label.set_color('b')
        ax.tick_params(axis='x', colors='b',labelsize=14)
        ax.tick_params(axis='y', colors='b',labelsize=14)

    

    # guarda la imagen
    if filename:
        plt.savefig(filename)
    

    plt.show()



def hist(img, size = 4, title = None, filename = None, axis = False):
    """Imprime la imagen y dibuja su histograma a un costado

    Args:
        img (numpy array): contiene la información de la imagen en un formato uint8
        size (int): define el tamaño de impresión de la gráfica
        title (str): título de la imagen
        filename (str): nombre del archivo donde se almacenará la imagen
        axis (bool): bandera para indicar si imprimen los ejes
    """

    f, c = img.shape[0:2]
    fig = plt.figure(figsize=(size*2, size))
    ax = fig.add_subplot(1,2,1)
    
    # IMPRESIÓN DE LA IMAGEN
    # verifica si la imagen está a color o escala de grises
    if len(img.shape) == 3: 
        im = ax.imshow(img,extent=None)
    else: 
        im = ax.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255)

    # coloca un título
    if title:
        ax.set_title(title,fontsize=14)


    # agrega escala
    if not axis:
        plt.axis('off')
    else:
        ax.grid(c='w')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w',labelsize=14)
        ax.tick_params(axis='y', colors='w',labelsize=14)


    # IMPRESIÓN DEL HISTOGRAMA
    ax = fig.add_subplot(1,2,2)
    # verifica si la imagen está a color o escala de grises
    if len(img.shape) == 3: 
        colors = ['r','g','b']
        for i,color in enumerate(colors):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            ax.plot(histr,c=color)
    else: 
        histr = cv.calcHist([img],None,None,[256],[0,256])
        ax.plot(histr, c='w',lw=2)
        ax.set_facecolor('k')
        ax.grid(alpha=0.3)


    # guarda la imagen
    if filename:
        plt.savefig(filename)
    

    plt.show()




def imgcmp(img1, img2, size = 4, title = None, filename = None, axis=False): 
    """Recibe dos imágenes y las imprime lado a lado para su comparación

    Args:
        img1 (numpy array): contiene la información de la imagen 1 en un formato uint8
        img2 (numpy array): contiene la información de la imagen 2 en un formato uint8
        size (int, optional): tamaño de la gráfica. Defaults to 4.
        title1 (str, optional): título de la imagen 1. Defaults to None.
        title2 (str, optional): título de la imagen 2. Defaults to None.
        filename (str, optional): _description_. Defaults to None.
    """

    title1 = None
    title2 = None

    if title is not None: 
        title1 = title[0]
        title2 = title[1]


    fig = plt.figure(figsize=(size*2, size))
    


    # IMAGEN 1
    ax = fig.add_subplot(1,2,1)
    if len(img1.shape) == 3: 
        im = ax.imshow(img1,extent=None)
    else: 
        im = ax.imshow(img1,extent=None,cmap='gray',vmin=0,vmax=255)

    # coloca un título
    if title1:
        ax.set_title(title1,fontsize=14)


    # agrega escala
    if not axis:
        plt.axis('off')
    else:
        ax.grid(c='w')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w',labelsize=14)
        ax.tick_params(axis='y', colors='w',labelsize=14)


    # IMAGEN 2
    ax = fig.add_subplot(1,2,2)
    if len(img2.shape) == 3: 
        im = ax.imshow(img2,extent=None)
    else: 
        im = ax.imshow(img2,extent=None,cmap='gray',vmin=0,vmax=255)

    # coloca un título
    if title2:
        ax.set_title(title2,fontsize=14)


    # agrega escala
    if not axis:
        plt.axis('off')
    else:
        ax.grid(c='w')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w',labelsize=14)
        ax.tick_params(axis='y', colors='w',labelsize=14)
    

    # guarda la imagen
    if filename:
        plt.savefig(filename)
    
    plt.show()








def splitrgb(img, filename=None):
    """Imprime la imagen y su alrededor imprime cada uno de sus colores por separado

    Args:
        img (numpy array): contiene la información de la imagen en un formato uint8
    """

    if len(img.shape) != 3: 
        return 0
    
    k = 8
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    
    fig = plt.figure(figsize=(k, k))

    # imagen a color
    ax = fig.add_subplot(2,2,1)
    ax.set_title("RGB") 
    im = ax.imshow(img,extent=None)
    plt.axis('off') # elimina  

    # rojo
    ax = fig.add_subplot(2,2,2)
    ax.set_title("R")
    im = ax.imshow(r,extent=None,cmap='gray',vmin=0,vmax=255)
    plt.axis('off')


    # verde
    ax = fig.add_subplot(2,2,3)
    ax.set_title("G")
    im = ax.imshow(g,extent=None,cmap='gray',vmin=0,vmax=255)
    plt.axis('off')

    # verde
    ax = fig.add_subplot(2,2,4)
    ax.set_title("B")
    im = ax.imshow(b,extent=None,cmap='gray',vmin=0,vmax=255)

    plt.axis('off')


    if filename != None: 
        fig.savefig(filename)

    plt.show()



def imgeq(img):
    """ Equalize a grayscale image
    Args:
        img (numpy array): Grayscale image to equalize
    Returns:
        eq (numpy array): Equalized image
    """
    cdf = imgcdf(img)[0]
    cdf_eq = []
    n = img.shape[0] * img.shape[1]
    m = min(i for i in cdf if i > 0)

    for i in cdf:
        if i >= m:
            cdf_eq.append(int(round(255*(i-m)/(n-m))))
        else:
            cdf_eq.append(0)
    eq = cv.LUT(img, np.array(cdf_eq).astype(np.uint8))
    return eq