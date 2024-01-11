from ultralytics import YOLO
import cv2
# Define los argumentos
model_path = "C:/Users/Usuario/Documents/Master/Vision/cnn/train2/weights/best.pt"
#model_path = "C:/Users/Usuario/Documents/Master/Vision/cnn/train2/weights/best.pt"
image_path = "C:/Users/Usuario/Documents/Master/Vision/Data/Cars0.png"
#source_path = 'best.pt'
image_path = "leon.jpg"

model = YOLO(model_path)
# Read an image using OpenCV
img = cv2.imread(image_path)
# Perform inference
results = model(img)
# View results
for result in results:
    # detection
    c = result.boxes.xywh.tolist()[0] # To get the coordinates.
    x, y, w, h = c[0], c[1], c[2], c[3] # x, y are the center coordinates.

    # Convert center x, y coordinates to top left x, y
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)

    # Ensure coordinates are within image bounds
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(img.shape[1], x1 + w))
    y2 = int(min(img.shape[0], y1 + h))

    # Crop the image
    cropped_img = img[y1:y2, x1:x2]

    # Display the cropped image
    cv2.imshow("Cropped Image", cropped_img)
    cv2.waitKey(0)  # Wait for a key press before closing the image window