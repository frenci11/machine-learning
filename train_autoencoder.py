import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, LeakyReLU
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
    
    input_img = Input(shape=(input_dim,))
    #encoder side
    with tf.device('/CPU:0'):
        encoded = Dense(1024)(input_img) 
        encoded = LeakyReLU(alpha=0.01)(encoded)
        encoded = Dense(512)(encoded)
        encoded = LeakyReLU(alpha=0.01)(encoded)
        #bottleneck reduction
        bottleneck= Dense(encoding_dim, name='bottleneck')(encoded)
        #decoder side
        decoded = Dense(512)(bottleneck)
        decoded = LeakyReLU(alpha=0.01)(decoded)
        decoded = Dense(1024)(decoded)
        decoded = LeakyReLU(alpha=0.01)(decoded)
        output = Dense(input_dim, activation='linear')(decoded)
        
        clear_session()  # Clear Keras/TensorFlow session
        #model compilation
        autoencoder_model = Model(inputs=input_img, outputs=output)
        autoencoder_model.summary()
        autoencoder_model.compile(optimizer='adam', loss='mse')

    
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
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