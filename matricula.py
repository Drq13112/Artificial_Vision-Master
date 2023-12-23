import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import math
import imutils
from sklearn.cluster import KMeans
from skimage.segmentation import clear_border
import imutils
import scipy.cluster.hierarchy as hcluster

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def Identificacion(query_img, train_img,features,thresh, margen):
    
    # Initialize lists
    list_kp1 = []
    list_kp2 = []

    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create(nfeatures=features)

    # Now detect the keypoints and compute
    # the descriptors for the query image
    # and train image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors, trainDescriptors)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 50 matches.
    img_final = cv2.drawMatches(query_img_bw,queryKeypoints,train_img_bw,trainKeypoints,matches, None, flags=2)
    plt.imshow(img_final)
    plt.show()

    # For each match...
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = queryKeypoints[img1_idx].pt
        (x2, y2) = trainKeypoints[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    
    # clustering
    clusters = hcluster.fclusterdata(list_kp1, thresh, criterion="distance")
    
    index = 0
    etiquetas=[]
    list_kp1_clus=[]
    for point in list_kp1:
        pos_x,pos_y = point
        list_kp1_clus.append([pos_x,pos_y,float(clusters[index])])
        etiquetas.append(clusters[index])
        index=index+1
    
    
    
    # Separar clusters
    clusters_separados={}
    for tupla in list_kp1_clus:
        cluster = tupla[2]
        if cluster not in clusters_separados:
            clusters_separados[cluster] = []
        clusters_separados[cluster].append([tupla[0],tupla[1]])

    # Calcula los centroides de cada cluster
    centroids = []
    for cluster in clusters_separados.values():
        centroids.append(np.mean(cluster, axis=0))

    # Cuenta cuántos puntos hay en cada clúster
    conteo_puntos_por_cluster = np.bincount(etiquetas)

    # Encuentra el índice del clúster con más puntos
    indice_cluster_mayor = np.argmax(conteo_puntos_por_cluster)
    puntos_cluster_mayor = []
    
    # Encuentra los puntos que pertenecen al clúster más grande
    for index in range(len(etiquetas)):
        if(etiquetas[index] == indice_cluster_mayor):
            puntos_cluster_mayor.append(list_kp1[index])

    # Convierte los puntos a un formato NumPy array
    puntos_array = np.array(puntos_cluster_mayor)
    
    # Encuentra las coordenadas mínimas y máximas para crear el bounding box con un margen
    x_min = round(min(puntos_array[:, 0]-margen))
    if(x_min < 0):
        x_min = 0
    y_min = round(min(puntos_array[:, 1]-margen/4))
    if(y_min < 0):
        y_min = 0
    x_max = round(max(puntos_array[:, 0]+margen))
    if(x_max > img.shape[1]):
        x_max = img.shape[1]
    y_max = round(max(puntos_array[:, 1]+margen/4))
    if(y_max > img.shape[0]):
        y_max = img.shape[0]

    point_color = (0, 0, 255)  # Red color in BGR

    # Define the point radius
    point_radius = 2

    # Loop through the points and draw them on the image
    # for point in puntos_array:
    #     coordinates = (int(point[0]), int(point[1]))
    #     cv2.circle(query_img, coordinates, point_radius,
    #                 point_color, -1)  # -1 fills the circle


    # Recorta la imagen usando las coordenadas del bounding box
    img_cropped = img[y_min:y_max, x_min:x_max]
    img_cropped.shape[0]
    img_cropped.shape[1]

    return img_cropped

if __name__ == "__main__":
    
    NombreRef = "./imagenes/Cars1.png"
    img = cv2.imread(NombreRef)
    # Añadir más placas de ejemplo y determinar el cluster que más coincida
    MatriculaRef = "./imagenes/matricula2.png"
    train = cv2.imread(MatriculaRef)
    col,fil,deep=img.shape
    ratio=fil/col
    # Especifica el nuevo tamaño (ancho, alto)
    nuevo_tamano = (round(640*ratio*3), 640)

    img_cropped = Identificacion(img, train,features=100,thresh=20,margen=70)

    # Aplica un desenfoque
    img_desenfocada_matr = cv2.GaussianBlur(img_cropped, (0, 0), 150)

    # Resta la imagen desenfocada de la imagen original para obtener la imagen nítida
    img_nitida_matr = cv2.addWeighted(img_cropped, 1.5, img_desenfocada_matr, -0.5, 0)

    # Especifica el nuevo tamaño para la imagen recortada (ancho, alto)
    nuevo_tamano_cropped = (img_nitida_matr.shape[1] * 2, img_nitida_matr.shape[0] * 2)

    # Redimensiona la imagen recortada
    img_cropped_ampliada = cv2.resize(img_nitida_matr, nuevo_tamano_cropped)

    # Show the final image
    cv2.imshow("Región de interés", img_cropped_ampliada)
    cv2.waitKey(0)

    placa = []

    
    # COMENZAMOS EL PROGRAMA
    flag = False
    flag1 = False
    opcion = 0
    #frames = []  # Vector para almacenar las imágenes del video
    image = None
    frame = None

    # Crear un objeto VideoWriter para almacenar el procesamiento del video
    #video = cv2.VideoWriter('videowriter.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (640, 480))

    # Almacenamiento de las imágenes proporcionadas en una matriz
    cars_imagenes = []

    for i in range(432):
        imagenCar = f"./imagenes/Cars{i}.png"
        imag = cv2.imread(imagenCar)


        if imag is None:
            print(f"No se pudo leer la imagen {imagenCar}")
            continue

        cars_imagenes.append(imag)

    
    car_original = img_cropped_ampliada.copy()
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

    gaussian = cv2.GaussianBlur(edges, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Dibujar contornos en una copia de la imagen original
    result = car_original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)

    # Mostrar la imagen con contornos
    cv2.imshow("Zonas Blancas Detectadas", result)

    for c in contours:

        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        epsilon = 0.02*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, epsilon, True)
  
        if len(approx)== 4 and area > 100:
        
            aspect_ratio = float(w)/h
            if 1.0 < aspect_ratio < 5.0:
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

                if len(pytesseract.image_to_string(warped, config='--psm 11')) < 3: break
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

    cv2.destroyAllWindows()
        