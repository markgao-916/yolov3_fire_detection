#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import time
import copy
import numpy as np
import pickle as pkl
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

LOAD_FROM_PKL = True

# print('tf.__version__: ', )

new_anno_lst = [[0], [1]]
# new_anno_lst = [[0], [1, 2]]

class YoloTest:
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        # model_dir = os.path.abspath(os.path.dirname(self.weight_file))
        # self.weight_file = os.path.join(model_dir, 'yolov3_test_loss=8.1157.ckpt-24')
        # self.weight_file = tf.train.latest_checkpoint(model_dir)
        # print(self.weight_file)

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        # with tf.name_scope('ema'):
        #     ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # self.sess = tf.Session()
        # self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        st = time.time()
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        pre_over_t = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )
        # predict_cost_T = time.time() - pre_over_t

        # print('pre_cost_T: %f, predict_cost_T: %f' %(pre_over_t - st, predict_cost_T))
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        time_total = 0.0
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        val_preds = []

        with open(self.annotation_path, 'r') as annotation_file:
            image_idx = 0
            for num, line in enumerate(annotation_file):
                st = time.time()
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                start = time.time()
                bboxes_pr = self.predict(image)
                time_total+=time.time() - start
                image_idx+=1
                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                pred_boxcses_per_img = []
                for bbox in bboxes_pr:
                    coor = list(np.array(bbox[:4], dtype=np.int32))
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    boxcs = []
                    boxcs.extend(coor)
                    boxcs.append(class_ind)
                    boxcs.append(score)
                    pred_boxcses_per_img.append(boxcs)
                val_preds.append(pred_boxcses_per_img)
                cost_T = time.time() - st
                # print('%d  cost_T: %f' % (num, cost_T))
                print(image_path)
        pkl_file = './pred_lst.pkl'
        fw = open(pkl_file, 'wb')
        pkl.dump(val_preds, fw, -1)
        fw.close()

    def voc_2012_test(self, voc2012_test_path):

        img_inds_file = os.path.join(voc2012_test_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        results_path = 'results/VOC2012/Main'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)
        pred_boxcses_per_img = []
        for image_ind in image_inds:
            image_path = os.path.join(voc2012_test_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)

            print('predict result of %s:' % image_ind)
            bboxes_pr = self.predict(image)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])




def analy():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    img_dir = os.path.join(root_dir, 'img')
    tmp_dir = os.path.join(root_dir, 'tmp')

    img_path_lst, gt_box_lst, label_lst = parse_anno(cfg.TEST.ANNOT_PATH)
    pkl_file = './pred_lst.pkl'

    pkl_fr = open(pkl_file, 'rb')
    pred_lst = pkl.load(pkl_fr)
    score_thresh_lst = list(np.arange(0.1, 1.0, 0.01))
    reg_flag = False
    num_cls = len(new_anno_lst)
    for thresh in score_thresh_lst:
        for idx_img in range(len(pred_lst)-1, -1, -1):
            boxcses = pred_lst[idx_img]
            num_box = len(boxcses)
            for idx_boxcs in range(num_box-1, -1, -1):
                boxcs = boxcses[idx_boxcs]
                score = boxcs[-1]
                for idx, anno in enumerate(new_anno_lst):
                    if boxcs[4] in anno:
                        boxcs[4] = idx
                        break
                if score < thresh:
                    del pred_lst[idx_img][idx_boxcs]

        recall_lst, prec_lst, pred_flag_lst = analy_rp(gt_box_lst, label_lst, pred_lst, num_cls=num_cls)
        recall = recall_lst[1]
        prec = prec_lst[1]
        if prec > 0.9 and not reg_flag:
            pred_lst_show = copy.deepcopy(pred_lst)
            pred_flag_lst_show = copy.deepcopy(pred_flag_lst)
            reg_flag = True

        print('score_thresh: %.2f, ' % thresh, end='')
        for i in range(num_cls):
            print('| cls_%d - recall: %.4f, prec: %.4f' % (i, recall_lst[i], prec_lst[i]), end='')
        print('')

    for idx_img, img_path in enumerate(img_path_lst):
        sub_dir, img_name = img_path.split('/')[-2:]
        img_name = img_name.split('.')[0]

        img = cv2.imread(img_path)

        #
        boxes_per_img  = gt_box_lst[idx_img]
        for box in boxes_per_img:
            box = list(map(int, box))
            img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (255, 0, 0))

        boxes_per_img = pred_lst_show[idx_img]
        pred_flag = pred_flag_lst_show[idx_img]
        for i, box in enumerate(boxes_per_img):
            color = (0, 255, 0) if pred_flag[i] else (0, 0, 255)
            box = list(map(int, box))
            img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), color)

        goal_name = '%s_%s.jpg' % (sub_dir, img_name)
        goal_path = os.path.join(tmp_dir, goal_name)
        cv2.imwrite(goal_path, img)


def analy_rp(gt_box_lst, label_lst, pred_lst, iou_thresh=0.3, num_cls=2):
    num_img_gt = len(gt_box_lst)
    num_img_pred = len(pred_lst)
    assert num_img_gt == num_img_pred, 'Error: num_img_gt != num_img_pred.'

    gt_per_cls_cnt = [0] * num_cls
    tp_per_cls_cnt = [0] * num_cls
    fp_per_cls_cnt = [0] * num_cls

    # counter the gt_box
    for labels in label_lst:
        for label in labels:
            gt_per_cls_cnt[label] += 1

    pred_flag_lst = []
    # analy per image
    for gt_box, labels, pred_boxcses in zip(gt_box_lst, label_lst, pred_lst):
        num_gt_box = len(gt_box)
        gt_matched_flag = [False] * num_gt_box
        num_pred_box = len(pred_boxcses)
        pred_flag = [True] * num_pred_box

        gt_box_arr = np.array(gt_box)
        for idx, boxcs in enumerate(pred_boxcses):
            pred_cls = boxcs[4]
            pred_score = boxcs[5]
            pred_box_ = np.array(boxcs[:4])

            ovr = IoU(pred_box_, gt_box_arr)
            max_iou = np.max(ovr)

            if max_iou < iou_thresh:
                pred_flag[idx] = False
                fp_per_cls_cnt[pred_cls] += 1
            else:
                arg_max = np.argmax(ovr)
                if gt_matched_flag[arg_max]:
                    pred_flag[idx] = False
                    fp_per_cls_cnt[pred_cls] += 1
                else:
                    if pred_cls == labels[arg_max]:
                        gt_matched_flag[arg_max] = True
                        tp_per_cls_cnt[pred_cls] += 1
                    else:
                        pred_flag[idx] = False
                        fp_per_cls_cnt[pred_cls] += 1

        pred_flag_lst.append(pred_flag)

    recall_lst = []
    prec_lst = []
    for i in range(num_cls):
        num_tp = tp_per_cls_cnt[i]  # sum(tp_per_cls_cnt)
        num_fp = fp_per_cls_cnt[i]  # sum(fp_per_cls_cnt)
        num_gt = gt_per_cls_cnt[i]  # sum(gt_per_cls_cnt)

        recall = float(num_tp) / float(num_gt)
        prec = float(num_tp) / float(num_tp + num_fp + 1e-5)
        recall_lst.append(recall)
        prec_lst.append(prec)
    return recall_lst, prec_lst, pred_flag_lst


def parse_anno(list_file):
    with open(list_file) as f:
        lines = f.readlines()
    img_path_lst = []
    gt_box_lst = []
    label_lst = []

    for line in lines:
        line = line.strip().split()
        img_path = line[0]
        boxes = line[1:]
        box = []
        label = []
        for bbox in boxes:
            lst = bbox.split(',')
            x = float(lst[0])
            y = float(lst[1])
            x2 = float(lst[2])
            y2 = float(lst[3])
            c = int(lst[4])
            for idx, anno in enumerate(new_anno_lst):
                if c in anno:
                    c = idx
                    break
            box.append([x, y, x2, y2])
            label.append(c)
        img_path_lst.append(img_path)
        gt_box_lst.append(box)
        label_lst.append(label)
    return img_path_lst, gt_box_lst, label_lst



def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (4, ): x1, y1, x2, y2
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


if __name__ == '__main__':
    test = YoloTest()
    test.evaluate()
    analy()



