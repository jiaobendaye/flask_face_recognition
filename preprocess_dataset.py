# coding: utf-8
'''
this file is to preocessing the images in IMAGES_DIR.
transform iamges to 128 embedding vector by using openface, and save as embedded.npy
the model is nn4_small2, 
my_model.h5 is created by model.save() and is free to use by keras.models.load_model()

https://github.com/krasserm/face-recognition

'''
import os

import cv2

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from align import AlignDlib

IMAGE_DIR = './images/'
LANDMARKER_FILE = './model/shape_predictor_68_face_landmarks.dat'
MODEL_FILE_H5 = './model/nn4_small2.h5'

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

def create_pretrain_model():
    nn4_small2 = load_model(MODEL_FILE_H5, custom_objects={'tf': tf})    
    alignment = AlignDlib(LANDMARKER_FILE)

    def align_image(img):
        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    metadata = load_metadata(IMAGE_DIR)
    embedded = np.zeros((metadata.shape[0], 128))

    for i, m in enumerate(metadata):
        img = load_image(m.get_image_path())
        img = align_image(img)
        # 数据规范化
        img = (img / 255.).astype(np.float32)
        # 获取人脸特征向量
        embedded[i] = nn4_small2.predict(np.expand_dims(img, axis=0))[0]
        print('Process', i, m, 'Finish')

    return embedded

if __name__ == "__main__":
    embedded = create_pretrain_model()
    np.save('./embedded.npy', embedded)

