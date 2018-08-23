import json
import os
import operator
import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify, url_for
from flask import request
from flask_cors import CORS
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import torch
import math
import numpy as np
import torchvision
import torch.nn as nn
import random, operator
from PIL import Image
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
# cuda = torch.cuda.is_available()

import shutil
from flask import Flask, render_template, request, make_response, jsonify

import torch
# from OpenSSL import SSL
# context = SSL.Context(SSL.SSLv23_METHOD)
# context.use_privatekey_file('yourserver.key')
# context.use_certificate_file('yourserver.crt')

app = Flask(__name__)
# sslify = SSLify(app)
CORS(app)
ALLOWED_EXTENSIONS = set(['mp3', 'wav'])

device = torch.device('cpu')
    


"""MODEL FRAMEWORK BEGIN"""


__all__ = ['ResNet', 'resnet']


model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : ResNet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18' : (BasicBlock, [2,2, 2,2]),
        '34' : (BasicBlock, [3,4, 6,3]),
        '50' : (Bottleneck, [3,4, 6,3]),
        '101': (Bottleneck, [3,4,23,3]),
        '152': (Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

#         x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc[0](x)
        x = self.fc[1](x1)
        x = self.fc[2](x)
        x2 = self.fc[3](x)
        x = self.fc[4](x2)
        x = self.fc[5](x)
        x3 = self.fc[6](x)
        out = self.fc[7](x3)

        return x2, out

def resnet(pretrained=False, depth=18, **kwargs):
    """Constructs ResNet models for various depths
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        depth (int) : Integer input of either 18, 34, 50, 101, 152
    """
    block, num_blocks = cfg(depth)
    model = ResNet(block, num_blocks, **kwargs)
    if (pretrained):
        print("| Downloading ImageNet fine-tuned ResNet-%d..." %depth)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet%d' %depth]))
    return model




""" MODEL FRAMEWORK END"""




""" ACTION TIME!! """
dataset_embeddings = pd.read_csv("firstEmbedN128.csv", header = None, engine='python')
# dataset_embeddings = pd.read_csv("C:/Users/spari/Projects/DATAMONK-june-july/PyTorch/Fresh/GTZANResearch/tripletDataCreation/CSVs/firstEmbedWSG.csv", header = None)


dataset_path = pd.read_csv("firstPathsN128.csv",sep= "/n", header = None, skipinitialspace = True, engine='python')
INPUT_SPECTROGRAM_FOLDER = "spectrograms"

# model = torch.load("finalModelCD")
# model = torch.load("resnet34N2")
genre_model = torch.load("resnet34N2", map_location=lambda storage, loc: storage).to(device)

# for param in model.parameters():
#     param.requires_grad = True
    
# model.fc.add_module("linear", nn.Linear(10, 10))
# # model.fc = fc_layers
print("recom model")



# if cuda:
#     model = model.cuda()
    # genre_model = genre_model.cuda()
    
    
UPLOAD_FOLDER = "upload"
SPECTRO_FOLDER = "spectrograms"
# shutil.rmtree(UPLOAD_FOLDER)
# shutil.rmtree(SPECTRO_FOLDER)  
        
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTRO_FOLDER'] = SPECTRO_FOLDER



# model.eval()


if not os.path.exists(UPLOAD_FOLDER):
     os.makedirs(UPLOAD_FOLDER)
        
        
        
if not os.path.exists(SPECTRO_FOLDER):
     os.makedirs(SPECTRO_FOLDER)

def check_allowed_file_format(filename):
    return filename.endswith("mp3")



GEN =  ['CLASSICAL','EDM','FOLK','HARD METAL','HIP HOP','JAZZ','POP','ROCK','ROMANTIC','SUFI']


def make_spectrograms_genre():
    import subprocess
    
    if not os.path.exists(SPECTRO_FOLDER):
                 os.makedirs(SPECTRO_FOLDER)
    subprocess.call('python make_spectrogram.py', shell=True)
    
    
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
     transforms.ToTensor(),
            normalize,    
     ])
    
    
    data = torchvision.datasets.ImageFolder(root= INPUT_SPECTROGRAM_FOLDER,transform=transform)

    dataloader = torch.utils.data.DataLoader(
        data,batch_size = len(data)
    )
    
    predicted_genre = None
    target_embed = None
    genre_model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data 
            # if cuda:
            #     images, labels  = images.cuda(), labels.cuda()

            emb,  outputs = genre_model(images)
            _, predicted = torch.max(outputs, 1)
            soft = nn.Softmax()            
            predicted_genre = (GEN[max(set((predicted)), key=list(predicted).count)])
            target_embed = soft(emb)
      
#     """ GENRE """
    possible_genre = [int(i) for i in predicted]
    from collections import Counter    
    poss = Counter(possible_genre)
    possible_genre = dict((GEN[int(key)], value) for (key, value) in poss.items())
    skeleton_dict = dict(zip(GEN, [0]*len(GEN)))
    skeleton_dict.update(possible_genre)
    final_dict = list(Counter(skeleton_dict).items())    
    final_dict.append(["GENRE", "COUNT"])
    
    
    final_dict.reverse()
    print("POSSIBLE", final_dict)
    return predicted_genre, final_dict


def make_spectrograms_recom():
    import subprocess
    
    if not os.path.exists(SPECTRO_FOLDER):
                 os.makedirs(SPECTRO_FOLDER)
    subprocess.call('python make_spectrogram.py', shell=True)
    
    
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
     transforms.ToTensor(),
            normalize,    
     ])
    
    
    data = torchvision.datasets.ImageFolder(root= INPUT_SPECTROGRAM_FOLDER,transform=transform)

    dataloader = torch.utils.data.DataLoader(
        data,batch_size = len(data)
    )
    
    predicted_genre = None
    target_embed = None
    genre_model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data 
            # if cuda:
            #     images, labels  = images.cuda(), labels.cuda()

            emb,  outputs = genre_model(images)
            _, predicted = torch.max(outputs, 1)
            soft = nn.Softmax()            
            predicted_genre = (GEN[max(set((predicted)), key=list(predicted).count)])
            target_embed = soft(emb)
      
#     """ RECOMMENDATIONS """
        
    embed = []
    
    for i in range(dataset_embeddings.shape[0]):
        embed.append(dataset_embeddings.iloc[i])    
    
    dist, ind = nearest_embeddings(embed, target_embed)
    top_5_song_score, genre_score, final_url = get_top_5_recom(dataset_path, ind)
    print(top_5_song_score)
    print(genre_score)
   
    return top_5_song_score, final_url
     


def nearest_embeddings(embed, target_embed):
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(np.array(list(embed), dtype=np.float))
    dist, ind = neigh.kneighbors(np.array((target_embed), dtype=np.float))
    return dist, ind

    
def get_top_5_recom(dataset_path, ind):
    
    genre_score = {}
    song_score = {}
    genre_noted = None
    song_noted = None

    for i in range(ind.shape[0]):
        for index in ind[i]:
            path = dataset_path[0][index]
            genre_noted = path.split("/")[3]
            song_noted = path.split("/")[4]

            if genre_noted in genre_score.keys():
                genre_score[genre_noted] = genre_score[genre_noted] + 1
            else:
                genre_score[genre_noted] = 1

            if song_noted in song_score.keys():
                song_score[song_noted] = song_score[song_noted] + 1
            else:
                song_score[song_noted] = 1
                
    top_5_song_score = sorted(song_score.items(), key=operator.itemgetter(1))[-10:]
    genre_score = sorted(genre_score.items(), key=operator.itemgetter(1))
    top_5_song_score.reverse()  
    final_suggest = []
    for i in range(len(top_5_song_score)):
        final_suggest.append(top_5_song_score[i][0])

    final_url = get_top_recm_url(final_suggest)
    # if final_url[0] == 0:
    #     final_url = final_url[1]
        
    return final_suggest, genre_score, final_url
                 
def get_top_recm_url(final_suggest):
    final_url = []
    try:
        for i in final_suggest:
            import urllib, time
            from googlesearch import search
            for url in search(i+" youtube", tld='com.pk', lang='es', stop=1, num = 1):
                url_key = url.split("=")[-1]
                main_url = urllib.parse.urljoin("https://www.youtube.com/embed/", url_key)
                time.sleep(0.1)
            final_url.append(main_url)        
        return final_url
    except:
        return [0, "Error! Server seems to be busy. Please try again!"]

    
@app.route("/recommendations", methods=['GET', 'POST'])
def recomm():        
        if request.method == 'GET':
            return "getting data"
        
        elif request.method == 'POST':
            file = request.files['audio_file1']
            if not file:
                return "No file"            
            if file.filename.split(".")[1] in ALLOWED_EXTENSIONS:
                shutil.rmtree(UPLOAD_FOLDER)
                shutil.rmtree(SPECTRO_FOLDER) 
                
                if not os.path.exists(UPLOAD_FOLDER):
                     os.makedirs(UPLOAD_FOLDER)
                        
                if file and check_allowed_file_format(file.filename):
                    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(save_path)
                    
                recom, urls  = make_spectrograms_recom()
                error = 0
                data = dict()                
                data['data'] = recom
                data['urls'] = urls
                if urls[0] == 0:
                    error = urls[1]
                data['error'] = error
                

            else:
                print("in")
                error = "Error! Check the input file type. Must be mp3."
                data = dict()
                data['error'] = error
            return json.dumps(data)
        

    
@app.route('/predictGenre', methods=['GET', 'POST'])
def predict_genre():
        if request.method == 'GET':
            return "getting data"
        
        elif request.method == 'POST':
            file = request.files['audio_file']
            if not file:
                return "No file"  
            if file.filename.split(".")[1] in ALLOWED_EXTENSIONS:

             
                shutil.rmtree(UPLOAD_FOLDER)
                shutil.rmtree(SPECTRO_FOLDER) 
                
                if not os.path.exists(UPLOAD_FOLDER):
                     os.makedirs(UPLOAD_FOLDER)
                        
                if file and check_allowed_file_format(file.filename):
                    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(save_path)
                    
                genre, possible_genres = make_spectrograms_genre()
                error = 0

                data = dict()
                data['error'] = error
                data['pydata'] = possible_genres
                data['genre'] =  genre

            else:
                # print("in")
                error = "Error! Check the input file type. Must be mp3."
                data = dict()
                data['error'] = error
            return json.dumps(data)
        
        
        
        
        
        
        
""" POPULARITY"""

def get_youtube_link(song, artist):
    from googlesearch import search
    try:
        for url in search(song + " "+ artist +" youtube", tld='com.pk', lang='es', stop=1, num = 1):
            main_url = url
        print(main_url)
        return main_url
    except:
        return "Error"
    
    

def get_features(name, artist_name = None):
    
    import pandas as pd 
    import spotipy 
    sp = spotipy.Spotify() 
    from spotipy.oauth2 import SpotifyClientCredentials 
    cid ="8b7bfb1e9128424590e7205d8cd810de" 
    secret = "b64cc0db77734701b557deb1f80d455b" 
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
    sp.trace=False 

    try:
        len_searchs = len(sp.search(name)['tracks']['items'])
    except:
        return 0, "Error! Server seems to be busy. Please try again!"
     

    name = name.replace("'", "").replace('’',"").replace(',',"").replace('&',"and").replace("?"," ").lower().split(" (")[0].split(".")[0].split(" -")[0]

        
    for j in range(len_searchs):
        try:
            if artist_name is not None:
                SPOTIFY_SEARCH = sp.search(name +" "+ artist_name)['tracks']['items'] 
                print("Got artist name as well")
            else:
                SPOTIFY_SEARCH = sp.search(name)['tracks']['items'] 
        except:
            return 0, "Error! This song is not in our database. Please try with another song."
        print("1")
        c_name = SPOTIFY_SEARCH[0]['name'] 
        print(c_name)
        
        c_name = SPOTIFY_SEARCH[j]['name']            
        current_name = c_name.lower().replace("'", "").replace('’',"").replace(',',"").replace("?"," ").replace('&',"and").split(" (")[0].split(".")[0].split(" -")[0]
        print("CURRENT",current_name, name)
            
        try:
            if name == current_name:            
                print(c_name)
                print("2")
                track_id = SPOTIFY_SEARCH[0]['id']
                popularity =SPOTIFY_SEARCH[0]['popularity']
                artist = SPOTIFY_SEARCH[0]['artists'][0]['name']
                curr_artist = artist.lower().replace("'", "").replace('’',"").replace(',',"").replace('&',"and").split(" (")[0].split(".")[0].split(" -")[0]
                
                print(curr_artist, artist_name)
                if artist_name == None:
                    artist_id = SPOTIFY_SEARCH[0]['artists'][0]['uri'].split(":")[2]
                    artist_popularity = sp.artist(artist_id)['popularity']

                    print("3")
                    echonest_features = sp.audio_features(track_id)[0]
                    year = SPOTIFY_SEARCH[0]['album']['release_date'].split("-")[0]
                    echonest_features['name'] = c_name
                    echonest_features['popularity'] = popularity
                    echonest_features['search_id'] = track_id
                    echonest_features['artist'] = artist
                    echonest_features['year'] = year
                    echonest_features['artist_popularity'] = artist_popularity
                    print("4")
                    audio_analysis = sp.audio_analysis(track_id)['track']
                    print("5")
                    echonest_features['duration'] = audio_analysis['duration']
                    echonest_features['end_of_fade_in'] = audio_analysis['end_of_fade_in']
                    echonest_features['start_of_fade_out'] = audio_analysis['start_of_fade_out']
                    
                    columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature', 'name', 'popularity', 'search_id', 'artist', 'year', 'artist_popularity','duration', 'end_of_fade_in', 'start_of_fade_out' ]
                    test_df = pd.DataFrame(columns = columns)
                    test_df =test_df.append(echonest_features, ignore_index=True)
                    test_df.drop(['type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms',
               'time_signature', 'name','search_id', 'artist'], axis = 1, inplace = True)
                    return test_df
                    
                
                if artist_name != None and curr_artist == artist_name: 
                    artist_id = SPOTIFY_SEARCH[0]['artists'][0]['uri'].split(":")[2]
                    artist_popularity = sp.artist(artist_id)['popularity']
                
                    print("3")
                    echonest_features = sp.audio_features(track_id)[0]
                    year = SPOTIFY_SEARCH[0]['album']['release_date'].split("-")[0]
                    echonest_features['name'] = c_name
                    echonest_features['popularity'] = popularity
                    echonest_features['search_id'] = track_id
                    echonest_features['artist'] = artist
                    echonest_features['year'] = year
                    echonest_features['artist_popularity'] = artist_popularity
                    print("4")
                    audio_analysis = sp.audio_analysis(track_id)['track']
                    print("5")
                    echonest_features['duration'] = audio_analysis['duration']
                    echonest_features['end_of_fade_in'] = audio_analysis['end_of_fade_in']
                    echonest_features['start_of_fade_out'] = audio_analysis['start_of_fade_out']
                    
                    columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature', 'name', 'popularity', 'search_id', 'artist', 'year', 'artist_popularity','duration', 'end_of_fade_in', 'start_of_fade_out' ]
                    test_df = pd.DataFrame(columns = columns)
                    test_df =test_df.append(echonest_features, ignore_index=True)
                    test_df.drop(['type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms',
               'time_signature', 'name','search_id', 'artist'], axis = 1, inplace = True)
                    print("TEST", test_df)
                    
                    
                    danceability = test_df['danceability'][0]*100
                    energy = test_df['energy'][0]*100
                    liveness = test_df['liveness'][0]*100
                    keys = [ 'energy', 'danceability', 'liveness']
                    extra_features = dict.fromkeys(keys, None)
                    extra_features['energy'] = ['energy', energy,'fill-color: blue ; fill-opacity: 0.3']
                    extra_features['danceability'] = ['danceability', danceability, 'fill-color: orange; fill-opacity: 0.5']
                    extra_features['liveness'] = ['liveness',liveness,'fill-color: green; ; fill-opacity: 0.5']
                    return test_df, list(extra_features.values())
        except:
            return 0, "Error! This song is not in our database. Please try with another song."
        
def predict(name, artist_name = None):
    test_df, plot_features= get_features(name, artist_name)
    # except:
    #     return 0, "Error! This song is not in our database. Please try with another song." , [], []
    # if test_df == 0:
    #     return test_df,plot_features,[],[]
    # print("TESTTTTTTTTTTTTT",test_df)
    
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.externals import joblib
    gridloadedr = joblib.load("Regxgboosttest0.pkl")
    gridloaded = joblib.load("xgboostgrid88.pkl")
    
    
    true_val = test_df['popularity']
    print("mm")
    inp = test_df.drop(["popularity"], axis = 1)
    print("in")
    prediction = gridloaded.predict(inp.astype("float"))
    print("uo")
    prediction_regressor = gridloadedr.predict(inp.astype("float"))    
    return true_val, prediction , prediction_regressor, plot_features 
    
    
    
@app.route('/predictPopularity', methods=['GET', 'POST'])
def predict_popularity():
        if request.method == 'GET':
            return "getting data"
        
        elif request.method == 'POST':
            song = request.form['song_name']
            artist = request.form['artist_name']
            if not song:
                return "No file"
#             if not artist:
#                 return "No file" 
            data = dict()
            if not artist:
                true_val, prediction, prediction_regressor, plot_features= predict(song) 
                print("ouch")
                if true_val == 0:
                    data['error'] = 0
                    data['error_desc'] = prediction 
                    return json.dumps(data)
            else:
                true_val, prediction, prediction_regressor, plot_features= predict(song, artist)
                print("pouch")
                # if true_val == 0:
                #     data['error'] = prediction 
                #     return json.dumps(data)
                # print("succ")
                          
            import urllib
            print(1)
            url = get_youtube_link(artist,song)
            # prit(2)
            url_key = url.split("=")[-1]
            print(3)
            data['error'] = 0
            print(4)
            if url_key == "Error":
                data['url'] = []
            data['url'] = urllib.parse.urljoin("https://www.youtube.com/embed/", url_key)
            print(5)
            data['true_val'] = str(true_val[0])
            print(6)
            data['predicted_class'] = str(prediction[0])
            print(7)
            data['predicted_regressor'] = str(prediction_regressor[0])
            print(8)
            data['plot_features'] = plot_features
            print(data)
                
            return json.dumps(data)
 
"""MUSIC COMPOSER """   
@app.route('/open-composer/')
def open_composer():
    return render_template('musicgeneration.html')   

@app.route('/open-predictor/')
def open_predictor():
    return render_template('popularitypredictor.html')   
    
        
@app.route('/genre-recom/')
def open_recom():
    return render_template('customsongrecomm.html')

@app.route('/')
def main():
    return render_template('main.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1',port='12344', 
        debug = True)

