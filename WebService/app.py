import json
import heapq
import librosa
import pandas as pd
import numpy as np
from flask import Flask
from keras.models import load_model
from predict_echo_feat import echonest_feature_maker


app = Flask(__name__)




## for function getAllSongs
# loading csv file (id-track)
df = pd.read_csv("id_track.csv",low_memory = False)


## for function get_distance
# reading the csv file (id-features)
echo =pd.read_csv("echo_features.csv", low_memory = False)  



## for function getFeatures
# keep the model loaded in the memory
model = load_model('acc_model.h5')


# helper function
def compute_distance(dft, b):
    # distance from 13K songs
    # uses min_heap concept
    h = []
    
    for i in range(dft.shape[0]):
        curr = dft.iloc[i]
        curr = curr.astype('float32')
        # distance calculatd here is always positive
        d = np.linalg.norm(curr-b)    

        if len(h) < 5:
            heapq.heappush(h, -d)

        else:
            # pop largest value present in current heap
            popped_value = heapq.heappushpop(h, -d)
   
    # make negative values positive again
    h = np.abs(h)   
    # sort and return list
    h.sort()    
    return h.tolist()





@app.route("/getAllSongs")
def getAllSongs():    
   
    
    idd = df.to_dict()['track_id']
    title = df.to_dict()['title']
    
    # creating list of dictionaries
    out = []
    for i in range(df.shape[0]):
        dict = {}
        dict['Id'] = idd[i]
        dict['Track'] = title[i]      
        out.append(dict)  
        
    # creating json         
    j = json.dumps(out)     
    return(j)  





@app.route("/computeDistance")
def get_distance(id, track): 
    
    # row of features of the given id
    
    ### NOTE - id requires to be of integer type!!!
    voi = echo.loc[echo['track_id'] == id]
    voi = voi.values
    
    # drop voi from the dataframe
    temp_df = echo.drop(voi.index.tolist()[0])    
  
    # gets list of minimum distances
    min_dist = compute_distance(temp_df, voi)
    
    # make json object
    d = json.dumps(min_dist)   
    
    return d
    
    
    
    
    
    
@app. route("/getFeature")
def upload_audio_features(audio):
    
## NEED TO CODE FOR THIS!!
    # enter your folder path"
#     base_path = ""
# path = path + audio

    
    voi = echonest_feature_maker(path,name,model)
    
    min_dist = compute_distance(echo, voi)
    
    return min_dist
    

    
    
    

if __name__ == '__main__':
    app.run(debug=True)