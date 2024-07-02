import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.models import Model
from collections import namedtuple
from sklearn.decomposition import PCA
import os


import tensorflow as tf
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model, load_model

    #input preprocessing
res=input_preprocessing(path_input)  
feature_vector= np.array(res.feature_vector)
feature_vector=feature_vector/255
print(f"Shape of feature_vectors: {feature_vector.shape}")

x_train, x_val = train_test_split(feature_vector, test_size=0.2, random_state=1)
print("x_train shape",x_train.shape,"x_val shape",x_val.shape)


# Define the autoencoder architecture
input_dim = x_train.shape[1]
encoding_dim = 512  # Number of features in the compressed representation
    
input_img = Input(shape=(input_dim,))
# Define the encoder model
encoder = Model(inputs=input_img, outputs=bottleneck)

# Define the decoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_hidden = autoencoder_model.layers[-4](encoded_input)  # 1024 layer
decoder_hidden = autoencoder_model.layers[-3](decoder_hidden)  # LeakyReLU
decoder_hidden = autoencoder_model.layers[-2](decoder_hidden)  # 2048 layer
decoder_hidden = autoencoder_model.layers[-1](decoder_hidden)  # LeakyReLU
decoder_output = autoencoder_model.layers[-1](decoder_hidden)  # Final dense layer

decoder = Model(inputs=encoded_input, outputs=decoder_output)

# Train the autoencoder with early stopping
history = autoencoder_model.fit(
    x_train, x_train,  # Input and target are the same for autoencoders
    epochs=100,
    batch_size=16,
    shuffle=True,
    validation_data=(x_val, x_val),  # Validation data
    callbacks=[early_stopping]
) 

# Predict using the autoencoder model
reconstructed_images = autoencoder_model.predict(feature_vector)
reconstructed_images = np.clip(reconstructed_images, 0, 1)   
reconstructed_images = reconstructed_images.reshape(-1, 112, 112, 3)

# Example of adding dropout layers for regularization
with tf.device('/GPU:0'):
    encoded = Dense(2048)(input_img) 
    encoded = LeakyReLU(alpha=0.01)(encoded)
    encoded = Dropout(0.5)(encoded)  # Add dropout
    encoded = Dense(1024)(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)
    encoded = Dropout(0.5)(encoded)  # Add dropout
    # Bottleneck reduction
    bottleneck = Dense(encoding_dim, name='bottleneck')(encoded)
    # Decoder side
    decoded = Dense(1024)(bottleneck)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dense(2048)(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dropout(0.5)(decoded)  # Add dropout
    output = Dense(input_dim, activation='linear')(decoded)

# Update model compilation with learning rate adjustment
autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
