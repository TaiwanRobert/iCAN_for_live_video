from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append('../../ican/tools')
sys.path.append('../../ican/lib')
sys.path.append('../../frcnn/tools')
sys.path.append('../../frcnn/lib')

import _init_paths
from PIL import Image
from ult.config import cfg
from utils.timer import Timer
import _init_paths
import pickle
import json
import numpy as np
import cv2
import multiprocessing as mp

import sys
import datetime
import mp3play
clip = mp3play.load('../../put_down_knife.mp3')

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush')


from Demo import run_ican
from Object_Detector import *

def end_pt(start_pt, text, font_size, count):
    y2 = start_pt[1] + 40
    x2 = start_pt[0] + 400
    
    return (int(x2), int(y2))

def create_text(img, text, shape, count, H_box):
    font_size = 1
    x1,y1, x2, y2 = H_box
    x1,y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
    start_pt = (x1+5, y1+5+30*(count-1))
    
    cv2.rectangle(img, start_pt, end_pt(start_pt, text, font_size, count), get_color(count),thickness = -1)
    cv2.putText(
        img, text, (x1 + 10, y1 -5 +(count*30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (255,255,255), 2, cv2.LINE_AA
    )
    
    
    
def get_color(number):
#     print('number:', number)
    num = str(int(number)%10)
    font_color = {
        '0': (100, 0, 0),
        '1': (0, 100, 0),
        '2': (0, 0, 100),
        '3': (100, 100, 0),
        '4': (100, 0, 100),
        '5': (0, 100, 100),
        '6': (100, 100, 100),
        '7': (200, 0, 0),
        '8': (0, 200, 0),
        '9': (0, 0, 200)
    }
    
    return font_color[num]

def create_bbox(img, box, count):
    x1,y1, x2, y2 = box
    x1,y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
#     print(x1,y1, x2, y2)
    
    cv2.line(img, (x1, y1), (x1, y2), get_color(count), 2)
    
    cv2.line(img, (x1, y2), (x2, y2), get_color(count), 2)
    
    cv2.line(img, (x2, y2), (x2, y1), get_color(count), 2)
    
    cv2.line(img, (x2, y1), (x1, y1), get_color(count), 2)
    
def print_image(Detection, im_data):
    img_shape = list(im_data.shape)
    new_shape = list(im_data.shape)
#     new_shape[1] = img_shape[1]+int(img_shape[1]*0.5)
    new_img = np.zeros(tuple(new_shape), np.uint8)
    new_img.fill(255)
    
    new_img[:img_shape[0],:img_shape[1]] = im_data

    HO_dic = {}
    HO_set = set()
    count = 0
    # print(Detection)
    action_count = -1
    had_knife, cut_obj_agent, hit_obj_agent = False, False, False
    for ele in Detection:
        
        H_box = ele['person_box'] 

        if tuple(H_box) not in HO_set:
            HO_dic[tuple(H_box)] = count
            HO_set.add(tuple(H_box))
            count += 1 

        show_H_flag = 0

        if ele['smile'][4] > 0.5:
            action_count += 1 
            show_H_flag = 1
            text = 'smile, ' + "%.2f" % ele['smile'][4]
            create_text(new_img, text, img_shape, action_count, H_box)
            

        if ele['stand'][4] > 0.5:
            action_count += 1 
            show_H_flag = 1
            text = 'stand, ' + "%.2f" % ele['stand'][4]
            create_text(new_img, text, img_shape, action_count, H_box)

        if ele['run'][4] > 0.5:
            action_count += 1 
            show_H_flag = 1
            text = 'run, ' + "%.2f" % ele['run'][4]
            create_text(new_img, text, img_shape, action_count, H_box)

        if ele['walk'][4] > 0.5:
            action_count += 1 
            show_H_flag = 1
            text = 'walk, ' + "%.2f" % ele['walk'][4]
            create_text(new_img, text, img_shape, action_count, H_box)
            
        if show_H_flag == 1:
            create_bbox(new_img, H_box, action_count)

        for action_key, action_value in ele.items():
            if ele['cut_obj_agent'] >= 0.9:
                cut_obj_agent = True
            if ele['hit_obj_agent'] >= 0.9:
                hit_obj_agent = True
                
            if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                if (not np.isnan(action_value[0])) and (action_value[5] > 0.05):
#                     print('active: ', CLASSES[np.int(action_value[4])])
                    O_box = action_value[:4]

                    action_count += 1

                    if tuple(O_box) not in HO_set:
                        HO_dic[tuple(O_box)] = count
                        HO_set.add(tuple(O_box))
                        count += 1      

                    create_bbox(new_img, H_box, action_count)
                    if CLASSES[np.int(action_value[4])] == 'knife':
                        had_knife = True
                    
                    text = action_key.split('_')[0] + ' ' + CLASSES[np.int(action_value[4])] + ', ' + "%.2f" % action_value[5]
                    create_text(new_img, text, img_shape, action_count, H_box)
                    create_bbox(new_img, O_box, action_count)
        if count >=2 and had_knife and cut_obj_agent and hit_obj_agent:
            print('放下屠刀立地成佛')
            clip.play()
            time.sleep(3)   
            clip.stop()
            

    return new_img 
    # cv2.imshow('frame', new_img[:,:,::-1])

# 選擇第二隻攝影機
cap = cv2.VideoCapture(0)
cap.set(3,420)
cap.set(4,320)

def main(frame, q):
    
    object_detection = run_frcnn(frame)
    print('after object_detection')
    tf.reset_default_graph()

    sess, hoi = run_ican(object_detection, frame)
    print('hoi: ', hoi)
    q.put(print_image(hoi, frame))

while(cap.isOpened()):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    '''
    q = mp.Queue()
    p = mp.Process(target=main,args=(frame, q))
    p.start()
    p.join()
    frame = q.get()
    '''
    # -----------------
    
    timer1 = Timer()
    timer1.tic()
    object_detection = run_frcnn(frame)
    print('object_detection: ', timer1.toc(average=False))
    print('-----------')
    #print('object_detection: ', object_detection)
    timer2 = Timer()
    timer2.tic()
    sess, hoi = run_ican(object_detection, frame)
    #print('hoi:', hoi)

    
    # -----------------
    timer3 = Timer()
    timer3.tic()
    frame = print_image(hoi, frame)
    print('print: ', timer3.toc(average=False))
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # sleep(5)

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()

sess.close()
frcnn_sess.close()

'''
{'cut_instr': 2,
 'snowboard_instr': 21,
 'cut_obj': 4,
 'surf_instr': 0,
 'skateboard_instr': 26,
 'kick_obj': 7,
 'eat_obj': 9,
 'carry_obj': 14,
 'throw_obj': 15,
 'eat_instr': 16,
 'smile': 17,
 'look_obj': 18,
 'hit_instr': 19,
 'hit_obj': 20,
 'ski_instr': 1,
 'run': 22,
 'sit_instr': 10,
 'read_obj': 24,
 'ride_instr': 5,
 'walk': 3,
 'point_instr': 23,
 'jump_instr': 11,
 'work_on_computer_instr': 8,
 'hold_obj': 25,
 'drink_instr': 13,
 'lay_instr': 12,
 'talk_on_phone_instr': 6,
 'stand': 27,
 'catch_obj': 28}
'''