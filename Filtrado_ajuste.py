import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
import pytesseract
from sklearn.cluster import KMeans
from skimage.segmentation import clear_border
import imutils


def Detecta_lineas(img, cantidad_lineas):

    output = cv2.merge([img, img, img])
    img = img[int(img.shape[0]/2):img.shape[0]]

    # Saco los contornos con Canny
    edges = cv2.Canny(img, 200, 250)

    # fijo el umbral a 0 porque sera la propia función la que encuentre el umbral idoneo
    umbral = 0

    # La variable bandera me sirve para coger solo la primera recta que detecte.
    # Lo hago para simplificar el resultado, además las rectas que no he tenido en cuenta
    # están muy cercanas entre sí, de modo que no importa si las desprecio.
    rectas = {}
    lines_1 = cv2.HoughLines(edges, 1, np.pi/180, umbral)
    Distancia = 0
    for i in range(len(lines_1)):

        # Auto gestion del umbral
        lines = cv2.HoughLines(edges, 2, np.pi/180, umbral)
        umbral = umbral+5

        if len(lines) > cantidad_lineas:
            print(f'Se han encontrado {len(lines)} rectas')

        else:

            # Dibujo y filtrado de rectas por secciones de la imagen
            print(f'Se han encontrado {len(lines)} rectas')
            for k in range(len(lines)):

                for rho, theta in lines[k]:
                    fil, col = img.shape
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(output, (x1, int(y1+fil)),
                             (x2, y2+int(fil)), (255, 0, 0), 2)

                    m = (y2-y1)/(x2-x1)
                    n = y1-m*x1
                    numero = str(k)
                    rectas[numero] = (m, n)

            plt.imshow(output)
            plt.show()

# Pongo que el limite de rectas son 40, para darle holgura al algoritmo,
# da igual que encuentre muchas rectas, luego las filtro y me quedo
# solo con la que me interesa


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
    # img_final = cv2.drawMatches(query_img_bw,queryKeypoints,train_img_bw,trainKeypoints,matches, None, flags=2)
    # plt.imshow(img_final)
    # plt.show()

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

    # Encuentra a qué clúster pertenece cada punto
    etiquetas = kmeans.labels_

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


def locate_license_plate_candidates(img, keep=5):

    # Convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perform a blackhat morphological operation that will allow
    # us to reveal dark regions (i.e., text) on light backgrounds
    # (i.e., the license plate itself)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    # self.debug_imshow("Blackhat", blackhat)
    cv2.imshow("Matches",  gray)
    cv2.waitKey(0)
    # compute the Scharr gradient representation of the blackhat
    # image in the x-direction and then scale the result back to
    # the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    # self.debug_imshow("Scharr", gradX)

    # blur the gradient representation, applying a closing
    # operation, and threshold the image using Otsu's method

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # self.debug_imshow("Grad Thresh", thresh)

    # perform a series of erosions and dilations to clean up the
    # thresholded image
    # Show the final image
    cv2.imshow("Matches", thresh)
    cv2.waitKey(0)
    kernel1 = (3, 3)
    kernel2 = (40, 40)
    thresh = cv2.erode(thresh, kernel1, iterations=1)
    thresh = cv2.dilate(thresh, kernel2, iterations=4)
    # self.debug_imshow("Grad Erode/Dilate", thresh)

    # take the bitwise AND between the threshold result and the
    # light regions of the imag
    #thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    #thresh = cv2.dilate(thresh, None, iterations=2)
    #thresh = cv2.erode(thresh, None, iterations=1)
    # self.debug_imshow("Final", thresh, waitKey=True)

    # find contours in the thresholded image and sort them by
    # their size in descending order, keeping only the largest ones
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

    # Show the final image
    cv2.imshow("Matches", thresh)
    cv2.waitKey(0)

    # initialize the license plate contour and ROI
    lpCnt = None
    roi = None
    candidates = cnts
    clearBorder = False
    minAR = 4
    maxAR = 5
    # loop over the license plate candidate contours
    for c in candidates:
        # compute the bounding box of the contour and then use
        # the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # check to see if the aspect ratio is rectangular
        if ar >= minAR and ar <= maxAR:
            # store the license plate contour and extract the
            # license plate from the grayscale image and then
            # threshold it
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Lo imprimo por pantalla
            imgcontoursRGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            imgcontoursRGB = cv2.drawContours(
                imgcontoursRGB, c, -1, (255, 0, 0), 5)
            cv2.imshow(' ', imgcontoursRGB)
            cv2.waitKey(0)

            # check to see if we should clear any foreground
            # pixels touching the border of the image
            # (which typically, not but always, indicates noise)
            if clearBorder:
                roi = clear_border(roi)

            # display any debugging information and then break
            # from the loop early since we have found the license
            # plate region
            break
            # return a 2-tuple of the license plate ROI and the contour
            # associated with it
    return (roi, lpCnt)


def test(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    blur = cv2.medianBlur(gray, 3)
    
    # edged = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)
    t_lower = 100 # Lower Threshold 
    t_upper = 200 # Upper threshold 
    aperture_size = 5 # Aperture size 
    L2Gradient = True # Boolean 
    
    # Applying the Canny Edge filter with L2Gradient = True 
    edged = cv2.Canny(blur, t_lower, t_upper, L2gradient = L2Gradient ) 
    
    laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)
    # Show the final image
    cv2.imshow("edges", sobelx)
    cv2.waitKey(0)
    cv2.imshow("edges", sobely)
    cv2.waitKey(0)
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # Hallamos el perímetro (cerrado) del contorno.
        perimeter = cv2.arcLength(c, True)

        # Aproximamos un polígono al contorno, con base a su perímetro.
        approx = cv2.approxPolyDP(c, .04 * perimeter, True)
        # Show the final image

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:

            # Lo imprimo por pantalla
            imgcontoursRGB = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
            imgcontoursRGB = cv2.drawContours(
                imgcontoursRGB, c, -1, (255, 0, 0), 5)
            cv2.imshow(' ', imgcontoursRGB)
            cv2.waitKey(0)

            # Calculamos la relación de aspecto.
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            print(aspect_ratio)

            if(aspect_ratio>=2.1 and aspect_ratio<=2.3):
                screenCnt = approx
                break

    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    # Show the final image
    cv2.imshow("edges", Cropped)
    cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def Centrar():

def centrar(img):

    # Primero suavizamos la imagen y luego la binarizamos con Otsu
    img_suavizada = cv2.medianBlur(img, 3)
    _, img_bin = cv2.threshold(
        img_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Obtengo las caracteristicas
    num_labels, img_labels, stats, cg = cv2.connectedComponentsWithStats(
        img_bin)

    # Defino el contorno de la calculadora
    contours, _ = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    ((centx, centy), (width, height), angle) = cv2.fitEllipse(cnt)

    # Lo imprimo por pantalla
    imgcontoursRGB = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    imgcontoursRGB = cv2.drawContours(
        imgcontoursRGB, contours, -1, (255, 0, 0), 5)
    cv2.imshow(' ', imgcontoursRGB)
    cv2.waitKey(0)

    # Giro la imagen de tal forma que la calculadora quede verticalmente
    M = cv2.getRotationMatrix2D((int(cg[1][0]), int(cg[1][1])), angle, 1)
    img_recortada = cv2.warpAffine(
        img, M, (img_bin.shape[0], img_bin.shape[1]))
    img_bin = cv2.warpAffine(img_bin, M, (img_bin.shape[0], img_bin.shape[1]))
    contours2, _ = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Recorto la imagen
    x, y, w, h = cv2.boundingRect(contours2[0])
    img_recortada = img_recortada[y:y+h, x:x+w]
    img_bin = img_bin[y:y+h, x:x+w]

    cv2.rotate(img, cv2.ROTATE_180)

    fil = img.shape[0]
    col = img.shape[1]
    """
    Segun que imagen son demasiado grandes para la pantalla del pc.
    De modo que reduzco la imagen a conveniencia de su tamaño.
    Para ello uso la funcion cv2.resize()
    """
    height = int(fil/2)
    width = int(col/2)
    dsize = (width, height)
    limg = cv2.resize(img, dsize)

    ptos1 = [(0, 0), (col, 0), (0, fil), (col, fil)]
    ptos1 = np.float32(ptos1)
    # ptos2 son los puntos donde mapeo ptos1. Es un rectangulo que empieza por
    # la esquina superior izda y va en sentido horario
    # ptos2 = np.float32([[200,125],[470,125],[470,315],[200,315]])

    # "Automatizo" las dimensiones de la imagen para que se ajuste adecuadamente a cualquiera imagen
    ptos2 = np.float32([[int(width/7), int(height/7)], [int(width-width/7), int(height/7)],
                       [int(width-width/7), int(height-height/7)], [int(width/7), int(height-height/7)]])

    # ------cv2.getPerspectiveTransform()
    M = cv2.getPerspectiveTransform(ptos1, ptos2)
    # ------cv2.warpPerspective()
    dst = cv2.warpPerspective(limg, M, (width, height))

    # Imprimo por pantalla
    cv2.imshow('salida', dst)
    cv2.waitKey(0)

    # Visualizar resultados con matplotlib
    plt.subplot(1, 2, 1), plt.imshow(limg), plt.title('Input')
    plt.axis(False)
    plt.subplot(1, 2, 2), plt.imshow(dst), plt.title('Output')
    plt.axis(False)
    plt.show()

    cv2.destroyAllWindows()
    cv2.getPerspective()
    cv2.warpPerspective()


# def Filtrado():


# def Deteccion_OCR():


if __name__ == "__main__":

    img = cv2.imread('../data/Cars322.png')
    # Añadir más placas de ejemplo y determinar el cluster que más coincida
    train = cv2.imread('../data/matricula2.png')

    img_cropped = Identificacion(img, train,features=300,cluster=10,margen=10)
    # Show the final image
    cv2.imshow("Matches", img_cropped)
    cv2.waitKey(0)
    # img_cropped = Identificacion(img_cropped, train,features=10, cluster=1,margen=10)
    # # Show the final image
    # cv2.imshow("Matches", img_cropped)
    # cv2.waitKey(0)

    # locate_license_plate_candidates(img_cropped)

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
