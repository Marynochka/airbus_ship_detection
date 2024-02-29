#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import random
import os
import numpy as np
import pandas as pd
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.models import Model
from tqdm import tqdm
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
from datetime import datetime
from metrics import dice_loss, dice_coef


END_IMAGE_SIZE = (256, 256, 3)
NUMBER = 10000 # size of sample for training
FRACTION_WITH_SHIPS = 0.75 #fraction of data with ships after undersampling
BATCH_SIZE = 16
EPOCH = 30

train_image_dir = 'airbus_ship_detection_unet/train_v2'
test_image_dir = "airbus_ship_detection_unet/test_v2"
# 1. Data preparation and Undersample Empty Images

# Read the CSV file containing ship segmentation information
masks = pd.read_csv("airbus_ship_detection_unet/train_ship_segmentations_v2.csv")
# Fill missing values in the 'EncodedPixels' column with empty strings
masks['EncodedPixels'] = masks['EncodedPixels'].fillna('')
# Group the data by 'ImageId' and concatenate the 'EncodedPixels' column values
masks = masks.groupby('ImageId')['EncodedPixels'].agg(' '.join).reset_index()

def train_ids(df, number=NUMBER, with_ships=FRACTION_WITH_SHIPS):
    """
    Split ImageIds into training ids, filtering out images with size less than 35 Kb.

    Args:
        df (pandas.DataFrame): The dataset containing the image information.
        number (int): The total number of ImageIds to be split.
        with_ships (float, optional): The proportion of ImageIds with ships, 0.75 by default.

    Returns:
        np.ndarray: The training ImageIds.
    """

    # Filter out images with size less than 35 Kb
    filtered_image_ids = []
    for img_id in df['ImageId'].unique():
        img_path = os.path.join(train_image_dir, img_id)  # Assuming train_image_dir is defined elsewhere
        if os.path.exists(img_path) and os.path.getsize(img_path) >= 35 * 1024:  # Convert Kb to bytes
            filtered_image_ids.append(img_id)

    # Filter the DataFrame based on the filtered image ids
    df = df[df['ImageId'].isin(filtered_image_ids)]

    # Randomly sample the filtered image ids with ships
    with_ships_num_train = int(number * with_ships)
    with_ships_df = list(df[df['EncodedPixels'] != '']['ImageId'].values)
    with_ships_df = random.sample(with_ships_df, with_ships_num_train)

    # Randomly sample the filtered image ids without ships
    without_ships_num_train = number - with_ships_num_train
    without_ships_df = list(df[df['EncodedPixels'] == '']['ImageId'].values)
    without_ships_df = random.sample(without_ships_df, without_ships_num_train)

    # Concatenate the sampled image ids and shuffle them
    ids = np.concatenate((with_ships_df, without_ships_df))
    np.random.shuffle(ids)

    return ids
#2. Mask and image encoding
def segmentation_mask(encoded_pixels: str, size=(768, 768)):
    """
    Generates a segmentation mask from the given encoded pixels.

    Args:
        encoded_pixels (str): The encoded pixels representing the mask.
        size (tuple, optional): The size of the mask, (768, 768) by default.

    Returns:
        np.ndarray: The segmentation mask.
    """
    # Initialize a mask with zeros
    mask = np.zeros(size[0] * size[1])

    # Split the encoded pixels and extract start pixels and lengths
    encoded_pixels = encoded_pixels.split()
    start_pixels = np.array([(int(x) - 1) for x in encoded_pixels[::2]])
    lengths = np.array([int(x) for x in encoded_pixels[1::2]])
    end_pixels = start_pixels + lengths

    # Fill the mask based on start and end pixels
    for i in range(start_pixels.shape[0]):
        mask[start_pixels[i]:end_pixels[i]] = 1

    # Reshape the mask to the specified size
    mask = mask.reshape(size).T
    return mask

def get_encoded_pixels_by_img_id(img_id, dataset):
    """
    Retrieves the encoded pixels for the given ImageId from the specific dataset.

    Args:
        img_id (str): The ImageId.
        dataset (pd.DataFrame): The dataset containing information for each image.

    Returns:
        str: The encoded pixels for the image ID.
    """
    # Retrieve the encoded pixels for the given ImageId
    encoded_pixels = dataset[dataset['ImageId'] == img_id]['EncodedPixels']
    
    # Check if the retrieved value is NaN (missing)
    if pd.isna(encoded_pixels.values):
        encoded_pixels = ' '  # If missing, assign an empty string
    
    # Join the encoded pixels into a single string
    encoded_pixels = ' '.join(encoded_pixels)
    
    return encoded_pixels

def fill_ds(ids, df, image_folder, img_size=(256, 256, 3)):
    """
    Fills the input and target arrays with the images and their corresponding segmentation masks by specific ids.

    Args:
        ids (list): The list of ImageIds.
        df (pd.DataFrame): The dataset containing information for each image.
        image_folder (str): The folder path where the images are stored.
        img_size (tuple, optional): The desired size of the images and masks, (256, 256, 3) by default.

    Returns:
        tuple: A tuple containing the input and target arrays.
    """
    # Initialize arrays for input and target data
    x = np.zeros((len(ids), img_size[0], img_size[1], img_size[2]), dtype=np.float32)
    y = np.zeros((len(ids), img_size[0], img_size[1], 1), dtype=np.float32)

    # Loop over ImageIds
    for n, img_id in tqdm(enumerate(ids), total=len(ids)):
        # Load and resize image
        img = np.asarray(Image.open(os.path.join(image_folder, img_id))
                         .resize((img_size[0], img_size[1])),
                         dtype=np.float32).reshape(img_size)
        x[n] = img / 255.0  # Normalize image data

        # Get encoded pixels for segmentation mask
        encoded_pixels = get_encoded_pixels_by_img_id(img_id, df)

        # Generate segmentation mask
        segmentation = segmentation_mask(encoded_pixels)
        segmentation = np.asarray(Image.fromarray(segmentation)
                                  .resize((img_size[0], img_size[1])),
                                  dtype=np.float32).reshape((img_size[0], img_size[1], 1))
        y[n] = segmentation

    return x, y


#3. Data initialisation
# Generate training ImageIds using train_ids function
train_ids = train_ids(masks, NUMBER, with_ships=FRACTION_WITH_SHIPS)

# Fill input and target arrays with images and segmentation masks using fill_ds function
X_data, y_data = fill_ds(train_ids, masks, train_image_dir, img_size=END_IMAGE_SIZE)

#4.U-NET model
def create_unet(filters=8,
                img_size=(256, 256, 3),
                dropout_rate=0.1,
                kernel_size=(3, 3),
                pool_size=(2, 2),
                strides=(2, 2)):
    """
    Creates a U-Net model for image semantic segmentation.

    Args:
        filters (int, optional): The number of filters in the first layer, 8 by default.
        img_size (tuple, optional): The input image size, (256, 256, 3) by default.
        dropout_rate (float, optional): The dropout rate, 0.1 by default.
        kernel_size (tuple, optional): The kernel size for convolutional layers, (3, 3) by default.
        pool_size (tuple, optional): The pool size for max pooling layers, (2, 2) by default.
        strides (tuple, optional): The strides for transpose convolutional layers, (2, 2) by default.

    Returns:
        keras.models.Model: The implemented U-Net model.
    """
    inputs = Input(img_size)

    # Contraction path
    c1 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D(pool_size)(c1)

    c2 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D(pool_size)(c2)

    c3 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D(pool_size)(c3)

    c4 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size)(c4)

    # Bridge (1024)
    c5 = Conv2D(filters * 16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(filters * 16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(filters * 8, pool_size, strides=strides, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(filters * 8, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(filters * 4, pool_size, strides=strides, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(filters * 4, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(filters * 2, pool_size, strides=strides, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(dropout_rate)(c8)
    c8 = Conv2D(filters * 2, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(filters, pool_size, strides=strides, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(dropout_rate)(c9)
    c9 = Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name='unet')
    print(model.summary())

    return model

#5. Defining training
def train(model, X_train, Y_train, validation_size=0.2, epochs=30, batch_size=16, model_name='unet_segmentation'):
    """
    Trains a given model on the provided training data.

    Args:
        model: The model to be trained.
        X_train: The input training data.
        Y_train: The target training data.
        validation_size (float, optional): The fraction of training data to be used for validation, 0.2 by default.
        epochs (int, optional): The number of training epochs, 30 by default.
        batch_size (int, optional): The batch size for training, 16 by default.
        model_name (str, optional): The name of the model, 'unet_segmentation' by default.

    Returns:
        keras.callbacks.History: The training history.
    """
    # Split the training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, random_state=42)

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss=[dice_loss], metrics=[dice_coef])

    # Get the current time for saving model checkpoints
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define callbacks for training
    callbacks = [
        CSVLogger("model_history_log.csv", append=True),
        ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6),
        ModelCheckpoint(f'models/{model_name}{current_time}.h5', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_dice_coef', patience=5, mode='max')
    ]

    # Train the model
    return model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# Create a UNet model
model = create_unet(filters=8)

# Train the model using training data and specified parameters
history = train(model, X_data, y_data, epochs=EPOCH, batch_size=BATCH_SIZE)

