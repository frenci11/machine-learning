import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from collections import namedtuple
from sklearn.decomposition import PCA
import os

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Function to visualize features
def VGG16_features(img_path, model, layer_name,visualize):
    
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    
    if width <= height:
        scale_factor = 224 / width
    else:
        scale_factor = 224 / height
        
    new_width = int(width * scale_factor)+1
    new_height = int(height * scale_factor)+1
    img = img.resize((new_width, new_height))
    
    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    right = left + 224
    bottom = top + 224
    
    img = img.crop((left, top, right, bottom))

    #img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Extract features using the specified layer
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_data)
    
    #----------this section is for visualization purposes only--------------------

    # Visualize the intermediate layer outputs
    if(visualize):
       
        plt.figure()
        plt.imshow(image.array_to_img(img_data[0]))
        plt.title("preprocess_input")
        
        num_filters = intermediate_output.shape[-1]
        size = intermediate_output.shape[1]
        
        print("numfilters",num_filters)
        print("filtersize",size)

        vis_filter=int(num_filters/8)
        vis_filter = min(vis_filter, 16)  # Limit the number of filters to visualize
        display_grid = np.zeros((size, size * vis_filter))
        
        for i in range(vis_filter):
            x = intermediate_output[0, :, :, i]
            x -= x.mean()
            x /= x.std() + 1e-5
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x
        
        scale = 100. / vis_filter
        plt.figure(figsize=(scale * vis_filter, scale))
        plt.title(f'Features from {layer_name}')
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig('conv1.png')
        plt.show()
    #------------------------------------------------------------------------------

    return intermediate_output.flatten()



def path_discovery(img_dir):
    """
    Discover image paths and corresponding labels in a directory.

    Parameters:
    img_dir (str): The root directory.

    Returns res(img_paths, dirs_visited, labels):
        - img_paths (list): List of full image file paths.
        - dirs_visited (list): List of directories visited.
        - labels (np.ndarray): Array of integer labels corresponding to each image's directory (from 1 to n).
    """
    discovery_results= namedtuple('discovery_results',['img_paths', 'dirs_visited', 'labels'])
    
    labels= np.array([],dtype=int)
    img_paths=[]
    dirs_visited=[]
    k=0
    for root, dirs, files in os.walk(img_dir):
        dirs_visited.append(root)
        for filename in files:
            img_path= os.path.join(root,filename)
            print(img_path)
            img_paths.append(img_path) 
            labels=np.append(labels,k)
        k=k+1
    
    return discovery_results(img_paths, dirs_visited, labels)


#pca component extraction mantaining 90% of test variance
def pca_extraction(feature_list, variance_percentage):
    """
    extract PCA component according to specified variance percentage

    Parameters:
    - feature_list
    - variance_percentage

    Returns results(pca,pca_result):
        - pca (class object)
        - pca_result (list) list of pca eigenvectors
    """
    results= namedtuple('results',['pca', 'pca_result'])
    
    if(variance_percentage<0 or variance_percentage>1):
        raise ValueError("variance_percentage must be between 0 and 1")
    pca= PCA()
    pca.fit(feature_list)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    pca_components=np.argmax(cumulative_explained_variance>variance_percentage)+1
    print(f"number of PCA components explaining {variance_percentage * 100}% of variance: {pca_components}")
    #applying pca with selected components number
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(feature_list)
    return results(pca,pca_result)