# show HOI detection

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from datasets.factory import get_imdb

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import pickle
from PIL import Image
import ipdb, glob

from nets.resnet_v1 import resnetv1

class ParseArgs(object):
    def __init__(self):
        self.img_dir = '../../iCAN/demo/'
        self.Demo_RCNN = '../../iCAN/demo/Object_Detection.pkl'
        self.img_format = 'png'



CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush')

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

# model path
tfmodel = cfg.ROOT_DIR + '/../ican/Weights/res50_faster_rcnn_iter_35000.ckpt'


if not os.path.isfile(tfmodel + '.meta'):
    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                   'our server and place them properly?').format(tfmodel + '.meta')) 
g1 = tf.Graph()

with g1.as_default():
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    frcnn_graph = tf.get_default_graph()
    frcnn_sess = tf.Session(config=tfconfig, graph = frcnn_graph)

    # load network
    net = resnetv1(num_layers=50)

    net.create_architecture("TEST", 81,
                      tag='default', anchor_scales=[4,8,16,32],anchor_ratios=[0.5,1,2])


    object_detection_saver = tf.train.Saver()
    object_detection_saver.restore(frcnn_sess, tfmodel)
print('after saver.restore')

def demo(frcnn_sess, net, im_file, RCNN):
    """Detect object classes in an image using pre-computed object proposals."""
    
#     image_name = im_file.split('/')[-1]
    tmp = []
    
    # Load the demo image
    im = im_file
    im = im[:, :, (2, 1, 0)]
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(frcnn_sess, net, im)
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
       
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        for det_inst in dets:
            if det_inst[4] > CONF_THRESH:
                inst_tmp = ['0000']
                if cls_ind == 1:
                    inst_tmp.append('Human')
                else:
                    inst_tmp.append('Object')
                inst_tmp.append(det_inst[:4])
                inst_tmp.append(np.nan)
                inst_tmp.append(cls_ind)
                inst_tmp.append(det_inst[4])
#                 print(inst_tmp)
                tmp.append(inst_tmp)
                    
                    
    return tmp
        
def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def run_frcnn(image_file):
    args = ParseArgs()
    RCNN = []
    count = 0
    RCNN = demo(frcnn_sess, net, image_file, RCNN)
    
    return RCNN

