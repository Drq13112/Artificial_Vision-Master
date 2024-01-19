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
import os


def Identificacion(query_img, train_img,features,thresh, margen):

    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    recortes = []

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
    clusters_min_size = {}

    for cluster, points in clusters_separados.items():
        if len(points) >= 10:
            clusters_min_size[cluster] = np.array(points)

    for cluster, points in clusters_min_size.items():
        # Encuentra las coordenadas mínimas y máximas para crear el bounding box con un margen
        x_min = round(min(points[:, 0]-margen))
        if(x_min < 0):
            x_min = 0
        y_min = round(min(points[:, 1]-margen/4))
        if(y_min < 0):
            y_min = 0
        x_max = round(max(points[:, 0]+margen))
        if(x_max > img.shape[1]):
            x_max = img.shape[1]
        y_max = round(max(points[:, 1]+margen/4))
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

        recortes.append(img_cropped)

    return recortes


if __name__ == "__main__":

    input_images_path = "C:/Users/Usuario/Documents/Master/Vision/esp_data"
    files_names = os.listdir(input_images_path)

    for file_name in files_names:
        image_path = input_images_path + "/" + file_name
        #print(image_path)
        img = cv2.imread(image_path)
        # Añadir más placas de ejemplo y determinar el cluster que más coincida
        train = cv2.imread("C:/Users/Usuario/Documents/Master/Vision/esp_data/Matricula.jpg")
        col,fil,deep=img.shape
        ratio=fil/col
        # Especifica el nuevo tamaño (ancho, alto)
        nuevo_tamano = (round(64*ratio*3), 64*3)
        # Redimensiona la imagen
        #img = cv2.resize(img, nuevo_tamano)

        img_cropped = Identificacion(img, train,features=300,thresh=40,margen=70)
        
        # Show the final image
        for image in img_cropped:
            cv2.imshow("Matches", image)
            cv2.waitKey(0)

