import keras
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Input, Dense,Conv2D, MaxPooling2D, Lambda,Input, Flatten,add, merge, concatenate 
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import random 
from PIL import Image
import operator
import time
from tqdm import tqdm
from keras.layers import Dense,Conv2D,ZeroPadding2D,Activation,BatchNormalization,MaxPooling2D, Lambda,Input, Flatten,add, merge, concatenate 

# get 128 embedding 
def mini_embed_model():
    inp =  Input(shape=(128, 1402, 1))
    x = Conv2D(64, (7,7), activation='relu', padding='same')(inp)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x) 
    
    x = Conv2D(64, (7,7), activation='relu')(inp)    
    x = MaxPooling2D((3,3))(x)
    
    x = Conv2D(128, (7,7), activation='relu', padding='same')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x) 
    
    x = Conv2D(128, (7,7), activation='relu')(x)
    x = MaxPooling2D((3,3))(x)
    
    x = Conv2D(256, (7,7), activation='relu', padding='same')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x) 
    
    x = Conv2D(256, (7,7), activation='relu')(x)
    x = MaxPooling2D((3, 3))(x)
    
    x = Flatten()(x)  
    x = Dense(512, activation='elu', name='fc1')(x)
    x = Dense(128, name='fc2')(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm')(x)
    return Model(inp, x)
    


# In[1]:


def embed_model():
    inp = Input(shape=(128, 1402, 1))
    x = ZeroPadding2D(padding=(3, 3), input_shape=(128, 1402, 1))(inp)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)    
    x = Lambda(lambda x: x ** 2)(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = Activation('relu')(x)
    
    x = Lambda(lambda x: x ** 2)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Conv2D(96, (1, 1))(x)
    x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Conv2D(96, (1, 1))(x)
    x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Conv2D(128, (3, 3))(x)
    x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Conv2D(16, (1, 1))(x)
    x = Activation('relu')(x)
    
    x = Flatten()(x)  
    x = Dense(512, activation='elu', name='fc1')(x)
    x = Dense(128, name='fc2')(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm')(x)
    return Model(inp, x)
        

    
    


# In[3]:


def triplet_loss(X, alpha = 0.2):
    a, p, n = X
    p_dist = K.sum(K.square(a - p), axis=-1)
    n_dist = K.sum(K.square(a - n), axis=-1)
    return K.sum(K.maximum(p_dist - n_dist + alpha, 0), axis=0)


# In[4]:


def identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)
    


# In[5]:

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def triplet_model(embed_model_type = True):

    
    # triplet inputs
    anchor_input   = Input(shape=(128, 1402, 1), name='anchor_input')
    positive_input = Input(shape=(128, 1402, 1), name='positive_input')
    negative_input = Input(shape=(128, 1402, 1), name='negative_input')
    
    # create embed model
    if embed_model_type:
        embeding_model = embed_model()
    else:
        embeding_model = mini_embed_model()
        
    # get embeddings
    anchor_embed = embeding_model(anchor_input)
    positive_embed = embeding_model(positive_input)
    negative_embed = embeding_model(negative_input)
    
    # merge  - depreciated 
    loss = merge([anchor_embed, positive_embed, negative_embed], mode= triplet_loss,  
                 name='loss', output_shape=(1, ))
    
    model = Model(input=[anchor_input, positive_input, negative_input], output=loss)
        
    model.compile(optimizer="adam", loss=identity_loss) 
#     model.compile(optimizer="rmsprop", loss=triplet_loss, metrics=[accuracy])
    
    return model
    
    
    
    
    

