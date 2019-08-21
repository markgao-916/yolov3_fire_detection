#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./pb_checkpoint/yolov3_test_loss=3.2609.ckpt-43.pb"
# image_path      = "/data/Fire/fire-detection/data/fire_smoke-0528/youtube_fire/Hazel_1775.jpg"
num_classes     = 2
input_size      = 416
graph           = tf.Graph()
metrics_path = '/home/amax/workspace-fire/Object-Detection-Metrics/samples/sample_2/test_detection'
name_map_dict={0:'Fire',1:'Smoke'}
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

def test_image():
    with tf.Session(graph=graph) as sess:
        val_txt_path = '/home/amax/workspace-fire/tensorflow-yolov3/data/my_data/fire_val_add_longmao.txt'
        with open(val_txt_path) as f:
            images = f.readlines()
        image_path_list = [i.strip().split(' ')[0] for i in images]
        for image_path in image_path_list:

            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.1)
            bboxes = utils.nms(bboxes, 0.5, method='nms')
            # image = utils.draw_bbox(original_image, bboxes)
            # image = Image.fromarray(image)
            # image.show()
            detection_txt = os.path.join(metrics_path,os.path.basename(image_path)).split('.')[0]+'.txt'
            txt_path = os.path.dirname(detection_txt)
            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
            with open(detection_txt,'w') as f:
                for bbox in bboxes:
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    x1,y1,x2,y2 = bbox[0:4]
                    writer_str = ' '.join([name_map_dict[class_ind], str(score), str(x1),str(y1), str(x2), str(y2)])
                    f.write(writer_str+'\n')

if __name__ == '__main__':
    test_image()

