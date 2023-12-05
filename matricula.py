import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

placa = []

# ARCHIVO DE CALIBRACIÓN
#ruta_calibracion = input("Ingrese la ruta del archivo de calibracion: ")
#calibracion = np.zeros((3, 3), dtype=np.float64)
#distorsion = np.zeros((1, 5), dtype=np.float64)

#try:
#    with open(ruta_calibracion) as file:
#        # Leer la primera línea del archivo
#        linea = file.readline()
#        numeros = list(map(float, linea.split()))
#        calibracion[0, :] = numeros[:3]
#        calibracion[1, :] = numeros[3:6]
#        calibracion[2, :] = numeros[6:]

        # Leer la segunda línea del archivo
#        linea = file.readline()
#        distorsion[0, :] = list(map(float, linea.split()))

        # Imprimir las matrices de calibración y distorsión
#        print("\nMatriz de calibracion:\n", calibracion, "\n")
#        print("Matriz de distorsion:\n", distorsion, "\n")

#except FileNotFoundError:
#    print("Error: no se pudo abrir el archivo.\n")


# COMENZAMOS EL PROGRAMA
flag = False
flag1 = False
opcion = 0
#frames = []  # Vector para almacenar las imágenes del video
image = None
frame = None

# Crear un objeto VideoWriter para almacenar el procesamiento del video
#video = cv2.VideoWriter('videowriter.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (640, 480))

# Almacenamiento de las imágenes ArUco proporcionadas en una matriz
cars_imagenes = []

for i in range(432):
    imagenCar = f"./imagenes/Cars{i}.png"
    imag = cv2.imread(imagenCar)


    if imag is None:
        print(f"No se pudo leer la imagen {imagenCar}")
        continue

    cars_imagenes.append(imag)

# Creamos los puntos de las esquinas a partir de una imagen como referencia

NombreRef = "./imagenes/Cars1.png"
car_original = cv2.imread(NombreRef)
cv2.imshow("Imagen original", car_original)


#filtrado
#escala de grises
car_gray = cv2.cvtColor(car_original, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagen grayscale", car_gray)

#suavizado NO
#suavizado = cv2.blur(car_gray, (4, 4))
#cv2.imshow("Imagen suavizado", suavizado)

#deteccion de bordes
edges = cv2.Canny(car_gray, 80, 240)
cv2.imshow("Imagen canny", edges)


cnts,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


for c in cnts:
  
  
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    epsilon = 0.02*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, epsilon, True)
  
    if len(approx)== 4 and area > 2000:
        
        aspect_ratio = float(w)/h
        if aspect_ratio > 2.4:
        # Transformación de perspectiva
            rect = cv2.minAreaRect(c)

            # Ajuste adicional para asegurar menor altura que anchura
            if rect[1][0] < rect[1][1]:
                rect = (rect[0], (rect[1][1], rect[1][0]), rect[2] - 90.0)

            box = cv2.boxPoints(rect)
        
            # Calcula la homografía para la transformación de perspectiva
            h, status = cv2.findHomography(np.array([box[1], box[2], box[3], box[0]]), np.array([[0, 0], [800, 0], [800, 200], [0, 200]]))
        
            # Aplica la transformación de perspectiva
            warped = cv2.warpPerspective(car_original, h, (800, 200))

            # Preprocesamiento en la región rectificada
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped = cv2.medianBlur(warped, 1)
            _, warped = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)

            if len(pytesseract.image_to_string(car_gray, config='--psm 11')) < 5: break
            else:
            # OCR en la región rectificada
                texto = pytesseract.image_to_string(warped, config='--psm 13')
                print('PLACA: ', texto)

                # Muestra la región rectificada en una nueva ventana
                cv2.imshow('Region Rectificada', warped)
                cv2.moveWindow('Region Rectificada', 780, 250)

                print('area = ', area)
                cv2.drawContours(car_original,[approx],0,(210,10,60), 2)
                cv2.putText(car_original, texto, (x - 20, y - 10), 1, 2.2, (0, 255, 0), 1)

      
cv2.imshow('Imagen completa', car_original)
cv2.moveWindow('Imagen completa', 45, 10)
cv2.waitKey(0)




#dilatación NO
#dilatado = cv2.dilate(edges, None)
#cv2.imshow("Imagen dilatado", dilatado)

#harris = cv2.cornerHarris(edges, 2, 3, 0.04)
 
#height, width = harris.shape
#color = (0, 255, 0)

#for y in range(0, height):
#    for x in range(0, width):
#        if harris.item(y, x) > 0.01 * harris.max():
#            cv2.circle(car_gray, (x, y), 1, (0, 255, 0), -1)

#cv2.imshow('Harris Result', harris)
#cv2.imshow('Harris Corner', car_gray)


#corners = cv2.goodFeaturesToTrack(car_gray, 60, 0.01, 10)
#corners = np.int0(corners)
   
#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(car_gray, (x, y), 1, (0, 255, 0), -1)
   
#cv2.imshow('corner', car_gray)

# Encontrar contornos
# Aplicar umbral para binarizar la imagen
_, thresh = cv2.threshold(car_gray, 200, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la imagen binarizada
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en una copia de la imagen original
result = car_gray.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen original y la imagen con contornos
cv2.imshow("Imagen Original", car_gray)
cv2.imshow("Zonas Blancas Detectadas", result)




#referencia_cars = np.array([[0, 0], [car.shape[1], 0], [car.shape[1], car.shape[0]], [0, car.shape[0]]], dtype=np.float32)

# Creamos una copia de la imagen para mostrar los puntos
car_con_puntos = car_original.copy()

# Dibujamos cada punto en la imagen con un círculo
#for punto in referencia_cars:
#    cv2.circle(car_con_puntos, tuple(punto), 10, (255, 0, 255), -1)

# Mostramos la imagen con los puntos en una ventana
#cv2.imshow("Imagen con puntos", car_con_puntos)
cv2.waitKey(0)
cv2.destroyAllWindows()
