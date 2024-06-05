import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import models

# Load the pre-trained model
model = models.load_model('improved_image_classifier.h5')

# Expected input size for the model
EXPECTED_SIZE = (32, 32)

# Class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Function to resize an image to the expected size with aspect ratio and padding
def resize_image(image, target_size):
    target_width, target_height = target_size

    # Get current image dimensions and aspect ratio
    height, width, _ = image.shape
    aspect_ratio = width / height

    # Determine new size while maintaining aspect ratio
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize to new dimensions
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)

    # Padding to fit exact target size
    top_padding = (target_height - new_height) // 2
    bottom_padding = target_height - new_height - top_padding
    left_padding = (target_width - new_width) // 2
    right_padding = target_width - new_width - left_padding

    # Apply padding with a constant border (black by default)
    padded_image = cv.copyMakeBorder(
        resized_image,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding
    )

    return padded_image


# Function to open a file dialog to select an image
def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    if file_path:
        img = cv.imread(file_path)  # Read image with OpenCV
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB
        img_resized = resize_image(img, EXPECTED_SIZE)  # Resize to 32x32
        show_image(img_resized)  # Display the image in the GUI
        return img_resized
    else:
        messagebox.showinfo("No image selected", "Please select an image to proceed.")
        return None


# Function to display the image in the GUI
def show_image(image):
    img_pil = Image.fromarray(image)  # Convert OpenCV image to PIL image
    img_tk = ImageTk.PhotoImage(img_pil)  # Convert PIL image to ImageTk
    image_label.configure(image=img_tk)  # Configure the label to display the image
    image_label.image = img_tk  # Keep a reference to prevent garbage collection


# Function to predict the class of the image
def predict_image():
    if selected_image is not None:
        # Normalize and prepare for model input
        img_for_model = np.array([selected_image / 255.0])  # Normalize
        prediction = model.predict(img_for_model)  # Make a prediction
        index = np.argmax(prediction)  # Get the index with the highest probability
        predicted_class = class_names[index]  # Retrieve the class name

        # Display the prediction in a new window
        prediction_window = tk.Toplevel(root)  # Create a new window for the prediction
        prediction_window.title("Prediction Result")
        prediction_label = tk.Label(
            prediction_window,
            text=f'Prediction: {predicted_class}',  # Display the prediction result
            font=("Arial", 16)
        )
        prediction_label.pack(pady=20)  # Add padding for aesthetics
    else:
        messagebox.showinfo("No image selected", "Please select an image before predicting.")


# Create the main window
root = tk.Tk()
root.title("Object Classifier")  # Set the title for the window

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Add buttons to select an image and predict
selected_image = None  # Global variable to hold the selected image


def handle_select_image():
    global selected_image
    selected_image = select_image()  # Select and resize the image

def main():
    select_button = tk.Button(root, text="Select Image", command=handle_select_image)
    select_button.pack(pady=10)  # Add padding for the button

    predict_button = tk.Button(root, text="Predict", command=predict_image)
    predict_button.pack(pady=10)  # Add padding for the button

    # Start the GUI event loop
    root.mainloop()  # Keep the application running
if __name__ == "__main__":
        main()