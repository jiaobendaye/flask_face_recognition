# coding: utf-8
import base64
from io import BytesIO
import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
from align import AlignDlib
import cv2

from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.keras.backend import set_session
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

THRESHOLD = 0.3
IMAGE_DIR = './images/'
MODEL_FILE_H5 = './model/nn4_small2.h5'
LANDMARKER_FILE = './model/shape_predictor_68_face_landmarks.dat'


global graph, nn4_small2
graph = tf.get_default_graph() 
set_session(session)
nn4_small2 = load_model(MODEL_FILE_H5, custom_objects={'tf': tf})
alignment = AlignDlib(LANDMARKER_FILE)



class IdentityMetadata(object):
    def __init__(self, base, name, file):
        self.base = base    # 数据集根目录
        self.name = name    # 目录名称
        self.file = file    # 文件名称

    def get_image_path(self):
        return os.path.join(self.base, self.name, self.file)

    def __repr__(self):
        return self.get_image_path()


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV默认使用BGR通道，转换为RGB通道
    return img[..., ::-1]

def rgb2gray(images):
    """将RGB图像转为灰度图"""
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])
##load database
embedded = np.load('embedded.npy')
metadata = load_metadata(IMAGE_DIR)
targets = np.array([m.name for m in metadata])



app = Flask(__name__)
app.debug = True

@app.route('/ping', methods=['GET','POST'])
def hello():
    return 'pong'

@app.route('/recognition', methods=['POST'])
def recognition():
    response = {'success': False, 'prediction': 'unk', 'debug': 'error'}
    
    received_image = False
    if request.method == 'POST':
        if request.files.get('image'): #图像文件
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
            print('receive file')
        elif request.get_json(): #base64 编码的图像文件
            encoded_image = request.get_json()['image']

            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            result =  _recognition(image)
            if result is not 'unk':
                response['prediction'] = result
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    print(response)
    return jsonify(response)
    

def test_model():
    img = cv2.imread('./test.jpg',1)[..., ::-1]
    img = align_image(img)
    img = (img / 255.).astype(np.float32)
    embedded = nn4_small2.predict(np.expand_dims(img, axis=0))[0]
    print(embedded[:5])


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

def _recognition(img):

    """
    return: result person name or unk
    """
    #preprocess
    img = align_image(img)
    if img is not None:
    # 数据规范化
        img = (img / 255.).astype(np.float32)
    else: return 'unk'
    # embedding
    global session
    global graph
    with graph.as_default():
        set_session(session)
        embedding = nn4_small2.predict(np.expand_dims(img, axis=0))
        all_distance = [distance(embedding, emb) for emb in embedded]
        min_dis =min(all_distance)
        print(min_dis)
        if min_dis > THRESHOLD:
            return 'unk'
        else:
            min_index = np.argmin(all_distance)
            return targets[min_index]

 