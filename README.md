()for image feature extraction, VGG16 convnet is used
()the fc2 layer is selected as output layer, so the network is truncated before the last layer
()the network outputs 4096 feature for every image
()the feature are then reduced with PCA to the 90% variance explained, so the vector numbers is not constant
()the feature are then passed to TSNE for 2D visualization
()the PCA feature are passed to Kmeans for training and clustering with 9 clusters.
