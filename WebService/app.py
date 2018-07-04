import json
import os
import operator
import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from collections import Counter
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
cuda = torch.cuda.is_available()

import shutil
from flask import Flask, render_template, request, make_response, jsonify

import torch

# from predict_echo_feat import echonest_feature_maker


app = Flask(__name__)
CORS(app)


UPLOAD_EXTENSIONS = set(['mp3', 'wav'])


    





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
        x = self.fc(x)

        return x

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


dataset_embeddings = pd.read_csv("firstEmbed.csv", header = None)
dataset_path = pd.read_csv("firstPaths.csv",sep= "\n",header = None)

genre_model = torch.load("Gtorch")

model = torch.load("Gtorch")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(57344, 128)

if cuda:
    model = model.cuda()
    genre_model = genre_model.cuda()
    
UPLOAD_FOLDER = "upload"
SPECTRO_FOLDER = "spectrograms"
# shutil.rmtree(UPLOAD_FOLDER)
# shutil.rmtree(SPECTRO_FOLDER)  
        
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SPECTRO_FOLDER'] = SPECTRO_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
     os.makedirs(UPLOAD_FOLDER)
        
if not os.path.exists(SPECTRO_FOLDER):
     os.makedirs(SPECTRO_FOLDER)

def check_allowed_file_format(filename):
    return filename.endswith("mp3")



GEN = ['dance','electronic','heavy_metal','hip-hop','jazz', 'rock','romantic', 'sufi']
def make_spectrograms():
    import subprocess
    
    if not os.path.exists(SPECTRO_FOLDER):
                 os.makedirs(SPECTRO_FOLDER)
    subprocess.call('python make_spectrogram.py', shell=True)
    
    
    transform = transforms.Compose([
                 transforms.Resize((127,223)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
    data = torchvision.datasets.ImageFolder(root="spectrograms",transform=transform)
    dataloader = torch.utils.data.DataLoader(data)
    
#     predicted_genre = None
#     all_pred = []
    genre_output = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            if cuda:
                images, labels  = images.cuda(), labels.cuda()
                
            outputs = genre_model(images)
            genre_output.append(outputs.cpu().data)
            print(torch.sigmoid(outputs))
#             _, predicted = torch.max(outputs.data, 1)
#             print(predicted)
#             predicted_genre = GEN[np.array(predicted)[0]]
#             print(predicted_genre)
#             all_pred.append(predicted_genre)
        genre_output = torch.cat(genre_output)
        _, max_indices = genre_output.max(dim=1)
        max_index = Counter(max_indices).most_common(1)[0] 
        print(max_index )
        
#     final_pred =Counter(all_pred).most_common(1)[0][0]
    
    print("********************************************", max_index)
    
            
        
    images = []
    for root, dir, filename in os.walk(SPECTRO_FOLDER):
         for file in filename:
#                 print(filename)
                filename=os.path.join(root, file).replace('\\', '/')
#                 print(filename)
                img = Image.open(filename)
                img = img.convert("RGB")
                img = img.resize((233,127))
                img = Variable( transform(img))
                images.append(img) 

        
    tranformed_spectrograms = torch.stack(images[40:],0) 
    if cuda:
            output = model(tranformed_spectrograms.cuda())

    target_embeddings = output.data.cpu().numpy()
    
    
    embed = []
    for i in range(dataset_embeddings.shape[0]):
        embed.append(dataset_embeddings.iloc[i])
      
    
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(np.array(list(embed), dtype=np.float))
    dist, ind = neigh.kneighbors(np.array(list(target_embeddings), dtype=np.float))
    
    distance = []
    indices = []
    for i in range(len(dist)):
        distance.extend(dist[i])
        indices.extend(ind[i])
        
    keys = indices
    values = distance
    data = dict(zip(keys, values))
    
    potential_indices = []
    
    for i in indices:
#         print(dataset_path.iloc[i][0].split("/")[3])
        if(dataset_path.iloc[i][0].split("/")[3] == predicted_genre):
            potential_indices.append(i)
        
    recom = {}

    for i in potential_indices:
        recom[dataset_path.iloc[i][0].split("/")[4]] = (data[i])
        if dataset_path.iloc[i][0].split("/")[4] in recom:
            recom[dataset_path.iloc[i][0].split("/")[4]] += (data[i])
            
    sorted_recom = sorted(recom.items(), key=lambda x: x[1])

    
    final_recom = []
    for i in range(len(sorted_recom)-1,len(sorted_recom)-6, -1):

        final_recom.append(sorted_recom.pop(i)[0])
    
    
    

        
    return final_recom, final_pred


    
@app.route("/recommendations", methods=['GET', 'POST'])
def recommendations():
         
       
        if request.method == 'GET':
            return "getting data"
        
        elif request.method == 'POST':
            file = request.files['audio_file']
            if not file:
                return "No file"
            
            
            shutil.rmtree(UPLOAD_FOLDER)
            shutil.rmtree(SPECTRO_FOLDER) 
            
            if not os.path.exists(UPLOAD_FOLDER):
                 os.makedirs(UPLOAD_FOLDER)

           

                    
            if file and check_allowed_file_format(file.filename):
                save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            
#             FOLDER_OF_CONCERN = os.path.join(UPLOAD_FOLDER, file.filename)
                
            file.save(save_path)
            recommendations, genre = make_spectrograms()
            
            data = dict()
            data['data'] = recommendations
            data['genre'] = genre
          
            
            
            
            return json.dumps(data)

@app.route('/')
def main():
    
    
  
 
    return render_template('action1.html')

if __name__ == '__main__':
    
    app.run(debug=True)
