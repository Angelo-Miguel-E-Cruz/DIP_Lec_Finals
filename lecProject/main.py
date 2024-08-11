# Import necessary packages
import cv2
import matplotlib.pyplot as plt
from rembg import remove
import tkinter as tk
from tkinter import filedialog


# Function to find the distance between rectangles in side_face and face
def is_close(rect1, rect2, threshold=100):
    # Calculate the center of each rectangle
    center1 = (rect1[0] + rect1[2] // 2, rect1[1] + rect1[3] // 2)
    center2 = (rect2[0] + rect2[2] // 2, rect2[1] + rect2[3] // 2)

    # Calculate the distance between the centers
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    # Return boolean
    return distance < threshold


# Function to remove image background
def remove_background(image):
    # Remove the background
    image_clear = remove(image)
    # Convert the background removed image to BGR format
    image_clear = cv2.cvtColor(image_clear, cv2.COLOR_RGBA2BGRA)
    return image_clear


# Function for face detection
def face_detection(fd_image):
    # Font for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Anti-aliasing to make text clearer
    antialiasing = cv2.LINE_AA
    if choice == "1":
        # Convert the image to BGR format
        fd_image = cv2.cvtColor(fd_image, cv2.COLOR_RGBA2BGRA)

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
        cv2.rectangle(fd_image, (x, y), (x + w, y + h), (0, 255, 0, 255), 2)
        cv2.rectangle(fd_image, (x - 1, y), (x + w + 20, y - 30), (0, 255, 0, 255), -1)
        cv2.putText(fd_image, 'Face', (x + 5, y - 10), font, 0.7, (0, 0, 0, 255), 1, antialiasing)

        # Get area of interest (face) and convert to gray
        aoi_color = fd_image[y:y + h, x:x + w]
        aoi_gray = cv2.cvtColor(aoi_color, cv2.COLOR_BGR2GRAY)

        # Detect eyes in area of interest (face)
        eyes = eye_cascade.detectMultiScale(aoi_gray)

        # Draw bounding rectangle for eyes rectangles
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(aoi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0, 255), 2)
            cv2.rectangle(aoi_color, (ex - 1, ey), (ex + ew + 10, ey - 30), (255, 0, 0, 255), -1)
            cv2.putText(aoi_color, 'Eye', (ex + 5, ey - 10), font, 0.5, (0, 0, 0, 255), 1, antialiasing)

    # Draw bounding rectangle for filtered_side_faces rectangles
    for (x, y, w, h) in filtered_side_faces:
        cv2.rectangle(fd_image, (x, y), (x + w, y + h), (0, 255, 0, 255), 2)
        cv2.rectangle(fd_image, (x - 1, y), (x + w + 10, y - 30), (0, 255, 0, 255), -1)
        cv2.putText(fd_image, 'Face', (x + 5, y - 10), font, 0.7, (0, 0, 0, 255), 1, antialiasing)

    return fd_image


print("\n\t\tBackground Remover and Face Detection\n")

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open file dialog to select the foreground image
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.gif")]
)

# Check if a file was selected
if not file_path:
    print("No file selected. Exiting.")
    exit()

# Read image and convert to grayscale
img = cv2.imread(file_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prompt the user to choose an option
print("Choose an option:")
print("1. Face Detection")
print("2. Background Removal")
print("3. Both Face Detection and Background Removal")
choice = input("Enter the number of your choice: ")

# Switch statement for user input
match choice:
    case "1":
        output = face_detection(img)
    case "2":
        output = remove_background(img)
    case "3":
        output = remove_background(img)
        output = face_detection(output)
    case _:
        print("Invalid Output")
        exit()

# Plot the image
plt.figure(figsize=(10, 10))
plt.imshow(output)
plt.axis('off')
plt.show()
