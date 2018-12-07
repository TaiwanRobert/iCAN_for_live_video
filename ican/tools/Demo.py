# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------

"""
Demo script generating HOI detections in sample images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb
import os
import os.path as osp

from ult.config import cfg
from models.test_demo import test_net

class ParseArgs(object):
    def __init__(self):
        self.iteration = 40000
        self.model = 'iCAN_ResNet50_VCOCO'
        self.prior_flag = 3
        self.object_thres = 0.4
        self.human_thres = 0.8
        self.img_dir = 'demo/'
        self.Demo_RCNN = '.....'
        self.HOI_Detection = '.....'
#     parser = argparse.ArgumentParser(description='Test an iCAN on VCOCO')
#     parser.add_argument('--num_iteration', dest='iteration',
#             help='Specify which weight to load',
#             default=300000, type=int)
#     parser.add_argument('--model', dest='model',
#             help='Select model',
#             default='iCAN_ResNet50_VCOCO', type=str)
#     parser.add_argument('--prior_flag', dest='prior_flag',
#             help='whether use prior_flag',
#             default=3, type=int)
#     parser.add_argument('--object_thres', dest='object_thres',
#             help='Object threshold',
#             default=0.4, type=float)
#     parser.add_argument('--human_thres', dest='human_thres',
#             help='Human threshold',
#             default=0.8, type=float)
#     parser.add_argument('--img_dir', dest='img_dir',
#             help='Please specify the img folder',
#             default='/', type=str)
#     parser.add_argument('--Demo_RCNN', dest='Demo_RCNN',
#             help='The object detection .pkl file',
#             default='/', type=str)
#     parser.add_argument('--HOI_Detection', dest='HOI_Detection',
#             help='Where to save the final HOI_Detection',
#             default='/', type=str)

#     args = parser.parse_args()
#     return args

def run_ican(object_detection, image):
    global prior_mask, Action_dic_inv, Action_dic, weight, tfconfig, ican_sess, saver,  ican_net
    args = ParseArgs()
    Demo_RCNN = object_detection
    try:
        print('not first')
        prior_mask
    except:
        print('first')
        with open( cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb" ) as gt:
            u_gt = pickle._Unpickler(gt)
            u_gt.encoding = 'latin1'
            prior_mask = u_gt.load()
        
        Action_dic = json.load(   open( cfg.DATA_DIR + '/' + 'action_index.json'))
        Action_dic_inv = {y:x for x,y in Action_dic.items()}
        
        weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
        print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
        # init session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        ican_sess = tf.Session(config=tfconfig)
        
        if args.model == 'iCAN_ResNet50_VCOCO':
            from networks.iCAN_ResNet50_VCOCO import ResNet50
        if args.model == 'iCAN_ResNet50_VCOCO_Early':
            from networks.iCAN_ResNet50_VCOCO_Early import ResNet50
        if args.model == 'iCAN_ResNet50_HICO':
            from networks.iCAN_ResNet50_HICO import ResNet50
        
        ican_net = ResNet50()
        ican_net.create_architecture(False)
    
        saver = tf.train.Saver()
        saver.restore(ican_sess, weight)

        print('Pre-trained weights loaded.')
    
    hoi = test_net(ican_sess, ican_net, Demo_RCNN, prior_mask, Action_dic_inv, image, args.HOI_Detection, args.object_thres,     args.human_thres, args.prior_flag)
    return ican_sess, hoi
