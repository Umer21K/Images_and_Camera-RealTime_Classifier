import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Data augmentation configuration
data_augmentation = ImageDataGenerator(
    rotation_range=15,  # Randomly rotate images by 15 degrees
    width_shift_range=0.1,  # Shift images horizontally
    height_shift_range=0.1,  # Shift images vertically
    shear_range=0.1,  # Shear images slightly
    zoom_range=0.1,  # Zoom in/out slightly
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill empty areas with nearest pixel
)

# Define the class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define the model with dropout and additional Conv2D layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # Input shape
    layers.MaxPooling2D((2, 2)),  # Max pooling
    layers.Conv2D(64, (3, 3), activation='relu'),  # Additional Conv2D layer
    layers.MaxPooling2D((2, 2)),  # Max pooling
    layers.Conv2D(128, (3, 3), activation='relu'),  # Another Conv2D layer
    layers.Flatten(),  # Flatten to feed into Dense layers
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(256, activation='relu'),  # Dense layer
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(10)  # Output layer for CIFAR-10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Set early stopping with a patience of 3 epochs
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with data augmentation for 10 epochs
history = model.fit(
    data_augmentation.flow(training_images, training_labels, batch_size=64),
    epochs=10,  # Train for 10 epochs
    validation_data=(testing_images, testing_labels),
    callbacks=[early_stopping]  # Early stopping to prevent overfitting
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Visualize training and validation accuracy over epochs
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Save the improved model
model.save('improved_image_classifier.h5')
