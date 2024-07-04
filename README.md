

(1)VGG16 model from keras is required, so install tensorflow
(2)autoencoder model is pretty heavy on the gpu, so if crashes, execute it on the cpu (instructions are on the code comments)
(3)after autoencoder subprocess starts, it cannot output to the STDOUT stream untill it finished, and the output is captured
in general it should take less than 10 min on CPU and less than 3 min on GPU
