import heapq
import json
import os

import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from keras.models import load_model

from predict_echo_feat import echonest_feature_maker

app = Flask(__name__)

upload_folder = '/Users/namakilam/workspace/Datamonk/MusicSims/DataMonk--Music-Similarity/WebService/music_uploads'

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

## for function getAllSongs
# loading csv file (id-track)
# df = pd.read_csv("id_track.csv",low_memory = False)
# tracks = df[['track_id', 'title']]
echonest_feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                            'tempo', 'valence']
columns = ['id', 'title']
columns.extend(echonest_feature_columns)


def load_tracks_data():
    if not os.path.isfile(upload_folder + '/tracks_data.csv'):
        return pd.DataFrame(columns=columns)
    else:
        return pd.read_csv(upload_folder + '/tracks_data.csv')[columns]


track_data = load_tracks_data()

last_track_id = track_data.size


# for index, row in tracks.iterrows():
#    track = dict()
#    track['id']=row['track_id']
#    track['title']=row['title']
#    track_data.append(track)
## for function get_distance
# reading the csv file (id-features)
# echo =pd.read_csv("echo_features.csv", low_memory = False)

def get_track_id():
    return int(track_data.shape[0]) + 1


## for function getFeatures
# keep the model loaded in the memory
model = load_model('acc_model.h5')


def check_allowed_file_format(filename):
    return filename.endswith('.mp3')


# helper function
def compute_distance(echonest_features, track_id):
    # distance from 13K songs
    # uses min_heap concept
    h = []

    for index, row in track_data.iterrows():
        if row['id'] == track_id:
            continue
        # distance calculatd here is always positive
        curr_feat = row[echonest_feature_columns].values
        d = np.linalg.norm(echonest_features - curr_feat)
        info = dict()
        info['distance'] = d
        info['track_id'] = row['id']
        info['title'] = row['title']
        h.append(info)

    return heapq.nsmallest(5, h, key=lambda s: s['distance'])


@app.route("/getAllSongs")
def getAllSongs():
    # idd = df.to_dict()['track_id']
    # title = df.to_dict()['title']
    data = dict()
    data['data'] = track_data[['id', 'title']].values.tolist()
    return jsonify(data)
    # resp = Response(js, status=200, mimetype='application/json')
    # return resp


@app.route("/computeDistance")
def get_distance():
    # row of features of the given id
    track_id = int(request.args.get('track_id'))
    ### NOTE - id requires to be of integer type!!!
    echo_features = track_data[track_data['id'] == track_id]
    echo_features = echo_features[echonest_feature_columns].values
    # gets list of minimum distances
    min_dist = compute_distance(echo_features, track_id)

    # make json object
    data = dict()
    data['title'] = track_data[track_data['id'] == track_id]['title'].values[0]
    data['data'] = min_dist
    d = json.dumps(data)

    return d


@app.route("/uploadFileComputeDistance", methods=['POST'])
def upload_audio_features():
    if request.method == 'POST':
        file = request.files['audio_file']
        if file and check_allowed_file_format(file.filename):
            save_path = os.path.join(upload_folder, file.filename)
            file.save(save_path)
            echonest_features = echonest_feature_maker(save_path, model)
            track_id = get_track_id()
            row = [track_id, file.filename]
            row.extend(echonest_features.tolist())
            track_data.loc[track_id] = row
            track_data.to_csv(upload_folder + '/tracks_data.csv')
            min_dist = compute_distance(echonest_features, track_id)
            data = dict()
            data['data'] = min_dist
            d = json.dumps(data)
            return d


if __name__ == '__main__':
    app.run(debug=False)
