# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:14:05 2020

@author: Rico
"""

import os
import cv2
import tensorflow as tf
import numpy as np
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
import joblib
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork


G3D_CATEGORIES = ["Clap", "Flap Arms", "Left Kick", "Left Punch", "Right Kick", "Right Punch", "Wave"]
UCF_11_CATEGORIES = ["basketball", "diving", "diving", "golf_swing", "horse_riding", "soccer_juggling", "swing", 
              "tennis_swing", "trampoline_jumping", "volleybal_spiking", "walking"]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Initialize computational graphs of both sub-networks
net_pose = HumanPoseIRNetwork()
net_gait = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 1)

# Load pre-trained models
net_pose.restore('models/MPII+LSP.ckpt')
net_gait.restore('models/H3.6m-GRU-1.ckpt')

logistic_model = joblib.load('models/logreg_g3d.sav')

def extract_features(video_frames):
    # Create features from input frames in shape (TIME, HEIGHT, WIDTH, CHANNELS) 
    spatial_features = net_pose.feed_forward_features(video_frames)

    # Process spatial features and generate temporal features 
    temporal_features, state = net_gait.feed_forward(spatial_features)
    return temporal_features

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

N = 70
last_N_frame = []

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    if len(last_N_frame) > N:
        temporal = extract_features(last_N_frame)
        prediction = logistic_model.predict([temporal])
        print("Predict Action:", G3D_CATEGORIES[prediction[0]])
        last_N_frame.clear()
    processed = imresize(frame, [299,299])
    last_N_frame.append(processed)
    key = cv2.waitKey(20)
    if key==27:
        break

cv2.destroyWindow("preview")