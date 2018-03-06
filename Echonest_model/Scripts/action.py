
# coding: utf-8

# In[2]:


import argparse
import pandas as pd
import numpy as np
from keras.models import load_model
# from extract_echonest_features import echonest_feature_maker


def predict():    
    model = load_model('acc_model.h5')
    test_data = pd.read_csv("input.csv", low_memory=False)
    
    test_data.drop([0,1], inplace = True)
    test_data.set_index("feature", inplace = True)
    
    test_data1 = np.expand_dims(test_data, axis=2)
    
    output = model.predict(test_data1)
    np.savetxt('output.csv', (output), delimiter=',')
    

def main():
    parser = argparse.ArgumentParser(description='Predicting echonest features.')
    parser.add_argument('location',metavar='location',type=str,help='Folder location of the songs')
#     parser.add_argument('csv name',metavar='csv_name',type=str,help='Name of the csv file')
    args=parser.parse_args()
    
    echonest_feature_maker.echonest_feature_maker(args.location,"input")
    predict()
    
    
if __name__ == "__main__":
    main()  
    
    
    

