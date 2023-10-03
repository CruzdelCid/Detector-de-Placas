# DETECTOR DE PLACAS 

El programa tiene el propósito de detectar los dígitos de las placas de Belgica. La detencción consiste en dos pasos
el primero la detección y recorte de la placa de la imagen y el segundo la detección de los dígitos y se predicción. 


Para la predicción de los dígitos se utilizó `XGBoost Classfier` con imágenes de ejemplo para se utilizaron imágenes 
de caracteres encontradas en la carpeta dataset.

## Requisitos 

- Python 3.9 
- Open CV 
- Pandas 
- scikit-learn (solo si se desea entrenar nuevamente el modelo)
- XGboost (solo si se desea entrenar nuevamente el modelo)


## Instrucciones 

Si ya se tiene todas las librerías instalas se puede ejecutar el siguiente comando: 

`python detector.py --p [path de la imagen]`

Ejemplo: 
`python detector.py --p C:\Users\chuzd\REPOS\ComputerVision\Proyecto1\LicencePlates\images65.jpg`


## Archivos disponibles
- dataset -> dataset de entrenamiento 
- LicensePlates -> imágenes con las placas de Bélgica
- modelos -> modelos entrenados 
- cvlib.py -> script de python para visualización 
- detector.py -> script del detector 
- proyecto_detector.ipynd -> notebook de pruebas del detector 
- pruebas_modelo.ipynd -> notebook usado para entrenar el modelo
