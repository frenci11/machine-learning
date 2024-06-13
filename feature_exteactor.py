import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

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

