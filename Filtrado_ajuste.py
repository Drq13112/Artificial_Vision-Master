import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
import pytesseract
from sklearn.cluster import KMeans
from skimage.segmentation import clear_border
import imutils
import scipy.cluster.hierarchy as hcluster


def Identificacion(query_img, train_img,features,cluster, margen):

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

    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(list_kp1)
    center = kmeans.cluster_centers_
    
    # clustering
    thresh = 15
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
    for point in puntos_array:
        coordinates = (int(point[0]), int(point[1]))
        cv2.circle(query_img, coordinates, point_radius,
                    point_color, -1)  # -1 fills the circle


    # Recorta la imagen usando las coordenadas del bounding box
    img_cropped = img[y_min:y_max, x_min:x_max]
    img_cropped.shape[0]
    img_cropped.shape[1]

    return img_cropped

# def Deteccion_OCR():


if __name__ == "__main__":

    img = cv2.imread('../data/Cars264.png')
    # Añadir más placas de ejemplo y determinar el cluster que más coincida
    train = cv2.imread('../data/matricula2.png')
    col,fil,deep=img.shape
    ratio=fil/col
    # Especifica el nuevo tamaño (ancho, alto)
    nuevo_tamano = (round(64*ratio*3), 64*3)
    # Redimensiona la imagen
    img = cv2.resize(img, nuevo_tamano)

    img_cropped = Identificacion(img, train,features=200,cluster=10,margen=70)
    # Show the final image
    # cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # B, G, R = cv2.split(img)
    # cv2.imshow('',B)
    # cv2.waitKey(0)
    # img_cropped = Identificacion(img_cropped, train,features=10, cluster=1,margen=10)
    # Show the final image
    cv2.imshow("Matches", img_cropped)
    cv2.waitKey(0)

    #locate_license_plate_candidates(img)

    #test(img_cropped)

    # Xml entrenado para la deteccion de autos
    # car_cascade = cv2.CascadeClassifier('cars.xml')

    # autos = car_cascade.detectMultiScale(img, 1.1, 1)

    # for (x, y, w, h) in autos:
    #     # Se  dibujan los rectangulos alrededor de los autos en formato rgb
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("video", img)

    # # Binarización de otsu tras un suavizado gaussiano
    # blur = cv2.GaussianBlur(img, (3, 3), 0)

    # #img_bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # # cv2.imshow("Img_bin",img_bin)
    # # cv2.waitKey()

    # img_bin = cv2.Canny(blur, 100, 200)
    # #img_bin = cv2.medianBlur(img_bin, 3)

    # cv2.imshow("Img_bin", img_bin)
    # cv2.waitKey()

    # Detecta_lineas(img_bin, 100)
    # img = cv2.imread('../data/matricula1.jpg', 0)
    # Detecta_lineas(img, 40)

    # """
    # Identificación matricula->centrar matrícula->filtrado->detección numeros
    # """

    # # cv2.imshow("Img_final",img_final)
    # # cv2.waitKey()
