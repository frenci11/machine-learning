import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import sys
import json
from collections import namedtuple
from utilities import input_preprocessing

def autoencoder_network(path_input):
    print("subprocess called\n")

    encoder_result= namedtuple('encoder_result',['hystory', 'autoencoder_model'])
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
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
    
    
    #encoder side
    with tf.device('/GPU:0'):
        input_layer_encoder = Input(shape=(input_dim,))
        hidden_enc = Dense(2048)(input_layer_encoder)
        hidden_enc = LeakyReLU(alpha=0.01)(hidden_enc)
        hidden_enc = Dropout(0.1)(hidden_enc)  # Add dropout
        hidden_enc = Dense(1024)(hidden_enc)
        encoded = LeakyReLU(alpha=0.01)(hidden_enc)

        # Bottleneck
        bottleneck = Dense(encoding_dim, name='bottleneck')(encoded)

        # Decoder
        input_layer_decoder = Input(shape=(encoding_dim,))
        hidden_dec = Dense(1024)(input_layer_decoder)
        hidden_dec = LeakyReLU(alpha=0.01)(hidden_dec)
        hidden_dec = Dropout(0.02)(hidden_dec)  # Add dropout
        hidden_dec = Dense(2048)(hidden_dec)
        hidden_dec = LeakyReLU(alpha=0.01)(hidden_dec)
        decoded = Dense(input_dim, activation='linear')(hidden_dec)

        # Define models
        encoder = Model(inputs=input_layer_encoder, outputs=bottleneck, name='encoder')
        encoder.summary()
        decoder = Model(inputs=input_layer_decoder, outputs=decoded, name='decoder')
        decoder.summary()

        # Autoencoder
        autoencoder_output = decoder(encoder(input_layer_encoder))
        autoencoder_model = Model(inputs=input_layer_encoder, outputs=autoencoder_output, name='autoencoder')

        # Compile the autoencoder model
        autoencoder_model.compile(optimizer='adam', loss='mse')
        autoencoder_model.summary()

    
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        print("autoencoder model fitting")
        # Train the autoencoder with early stopping
        history = autoencoder_model.fit(
            x_train, x_train,  # Input and target are the same for autoencoders
            epochs=100,
            batch_size=16,
            shuffle=True,
            validation_data=(x_val, x_val),  # Validation data
            callbacks=[early_stopping]
        ) 
    
    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('autoencoder_val_loss.png')
    #plt.show()
    
    #return encoder_result(history,autoencoder_model)
    print("saving model")
    autoencoder_model.save('autoencoder_model')
    print("saved")



if __name__ == '__main__':
    img_paths = sys.argv[1:]
    autoencoder_network(img_paths)