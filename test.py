from tensorflow import keras
import seaborn as sns
import numpy as np
import pandas as pd
from metrics import *
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


custom_objects = {'dice_loss': dice_loss,
                 'dice_coef': dice_coef}

# Load the model using the custom object scope
model = keras.models.load_model('models/unet_segmentation2024-02-29_23-46-48.h5', custom_objects=custom_objects)

test_imgs = ['00dc34840.jpg', '00c3db267.jpg', '00aa79c47.jpg', '00a3a9d72.jpg']
test_image_dir = "airbus_ship_detection_unet/test_v2"

def preprocess_image(image_path, target_size=(256, 256)):
    try:
        image = load_img(image_path, target_size=target_size)
        image_array = img_to_array(image) / 255.0  # Normalize pixel values
        return image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Define a function to make predictions
def predict_image(image_array):
    try:
        prediction = model.predict(np.expand_dims(image_array, axis=0))
        return prediction[0]  # Assuming prediction is a numpy array
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# Plot original image and its prediction
fig, axes = plt.subplots(len(test_imgs), 2, figsize=(12, 6*len(test_imgs)))

for i, image_name in enumerate(test_imgs):
    # Load and preprocess the image
    image_path = os.path.join(test_image_dir, image_name)
    image_array = preprocess_image(image_path)
    if image_array is None:
        continue
    
    # Make predictions
    prediction = predict_image(image_array)
    if prediction is None:
        continue
    
    # Plot original image
    original_image = image_array * 255.0  # Convert back to original scale
    axes[i, 0].imshow(original_image.astype(np.uint8))
    axes[i, 0].set_title("Original Image")
    
    # Plot prediction
    axes[i, 1].imshow(prediction, cmap='Blues_r')  # Assuming prediction is a grayscale image
    axes[i, 1].set_title("Predicted Mask")

plt.tight_layout()
plt.show()