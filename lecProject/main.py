import cv2
import matplotlib.pyplot as plt
from rembg import remove
import numpy as np

# Function to find the distance between rectangles in side_face and face
def is_close(rect1, rect2, threshold=100):
    # Calculate the center of each rectangle
    center1 = (rect1[0] + rect1[2] // 2, rect1[1] + rect1[3] // 2)
    center2 = (rect2[0] + rect2[2] // 2, rect2[1] + rect2[3] // 2)

    # Calculate the distance between the centers
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return distance < threshold


# Read image and make gray
img = cv2.imread("fruits.jpg")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Remove the background
image_no_bg = remove(img)

# Convert the background removed image to a format that OpenCV can work with
image_no_bg = cv2.cvtColor(np.array(image_no_bg), cv2.COLOR_RGBA2BGRA)

# cv2 code to detect faces in image
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# cv2 code to detect eyes in image
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Detect faces
faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
side_faces = side_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Filter side_faces to remove ones that are close to face
filtered_side_faces = []
for side_face in side_faces:
    # Only add side_faces that are not close to face
    if not any(is_close(side_face, face_rect) for face_rect in faces):
        filtered_side_faces.append(side_face)

# Draw bounding rectangle for face rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(image_no_bg, (x, y), (x + w, y + h), (0, 255, 0, 255), 2)

    # Get area of interest (face) and convert to gray
    aoi_color = img[y:y + h, x:x + w]
    aoi_gray = cv2.cvtColor(aoi_color, cv2.COLOR_BGR2GRAY)

    # Detect eyes in area of interest (face)
    eyes = eye_cascade.detectMultiScale(aoi_gray)

    # Draw bounding rectangle for eyes rectangles
    for (ex, ey, ew, eh) in eyes:
        # Map the eye coordinates back to the original image coordinates
        cv2.rectangle(image_no_bg, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0, 255), 2)

# Draw bounding rectangle for filtered_side_faces rectangles
for (x, y, w, h) in filtered_side_faces:
    cv2.rectangle(image_no_bg, (x, y), (x + w, y + h), (0, 255, 0, 255), 2)

# Plot the image
plt.figure(figsize=(20, 10))
plt.imshow(image_no_bg)
plt.axis('off')
plt.show()
