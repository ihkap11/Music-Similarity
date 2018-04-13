
# coding: utf-8

# In[2]:
from PIL import Image
import numpy as np
from keras.preprocessing import image

def genre_count_dataset(data, rang):
    cl = h = m = r = p = co = 0
    
    for i in range(rang):    
        if data['Genre'][i] == "Classical":
            cl += 1
        if data['Genre'][i] == "Hip-Hop":
            h += 1
        if data['Genre'][i] == "Metal":
            m += 1
        if data['Genre'][i] == "Rock":
            r += 1
        if data['Genre'][i] == "Pop":
            p += 1
        if data['Genre'][i] == "Country":
            co += 1
    
    print("Classical", cl)   
    print("Hip-Hop", h) 
    print("Metal", m)
    print("Rock", r) 
    print("Pop", p) 
    print("Country", co) 
    
    return cl, h ,m ,r ,p ,co


def get_model_memory_usage(batch_size, model):
   import numpy as np
   from keras import backend as K

   shapes_mem_count = 0
   for l in model.layers:
       single_layer_mem = 1
       for s in l.output_shape:
           if s is None:
               continue
           single_layer_mem *= s
       shapes_mem_count += single_layer_mem

   trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
   non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

   total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
   gbytes = np.round(total_memory / (1024.0 ** 3), 3)
   return gbytes


# In[4]:

def img_from_ID(data, id):
    return data.iloc[id-1]['Images']


def images_from_ids(data):
    from PIL import Image
    import numpy as np
    from keras.preprocessing import image
    
    images = []
    for id in range(1,data.shape[0]+1):
        curr_path = data.iloc[id-1][' Spectrogram']

        img = Image.open(curr_path).convert('L')
        img = img.resize((1402, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #     print(x.shape) --> (1, 128, 1402, 1)        
        images.append(x)
    data['Images'] = images 
    print("Images added to dataframe")
    return data