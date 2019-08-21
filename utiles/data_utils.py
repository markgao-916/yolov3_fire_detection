# coding: utf-8

from __future__ import division, print_function

import numpy as np
import cv2
import sys
from utiles.data_aug import *
import random

PY_VERSION = sys.version_info[0]
iter_cnt = 0

DEBUG = False

def gen_v3_anchor_lst(anchors, grid_size_lst):
    anch_lst = list(anchors)
    stride_lst = [32, 16, 8]
    # grid_size_lst = [13, 26, 52]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    v3_anchor_lst = []
    for idx, mask_branch in enumerate(anchors_mask):
        stride = stride_lst[idx]
        grid_size = grid_size_lst[idx]
        anchor_branch = []
        for mask in mask_branch:
            cell_size = np.array(anch_lst[mask])
            anch = generate_anchors(grid_size, grid_size, cell_size, stride, 1)
            # anch = np.reshape(anch, [-1, 4])

            # anch_centers = (anch[:, 0:2] + anch[:, 2:4]) / 2
            # anch_sizes = anch[:, 2:4] - anch[:, 0:2]

            # anch_xywh = np.concatenate([anch_centers, anch_sizes], axis=-1)

            anchor_branch.append(anch)
            # print('anch:', anch)
        # anchor_branch = np.array(anchor_branch)
        # anchor_branch = np.reshape(anchor_branch, [-1, 4])
        v3_anchor_lst.append(anchor_branch)
    return v3_anchor_lst

def gen_anchor_lay(anchors, map_idx, k, grid_size):
    anch_lst = list(anchors)
    stride_lst = [32, 16, 8]
    # grid_size_lst = [13, 26, 52]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    mask = anchors_mask[map_idx][k]
    cell_size = np.array(anch_lst[mask])
    stride = stride_lst[map_idx]

    anch_lay = generate_anchors(grid_size, grid_size, cell_size, stride, 1)

    return anch_lay


def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    line_idx = int(s[0])
    pic_path = s[1]
    s = s[2:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels

def generate_anchors(grid_height, grid_width, cell_size, stride, scale=1.0):
    """

    :param grid_height:
    :param grid_width:
    :param cell_size: [W, H]
    :param stride:
    :param scale:
    :return:
    """
    inverse_scale = 1. / scale
    y_centers = np.arange(grid_height) * stride
    y_centers = (y_centers + cell_size[1] / 2) * inverse_scale
    x_centers = np.arange(grid_width) * stride
    x_centers = (x_centers + cell_size[0] / 2) * inverse_scale
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)
    bboxes_centers = np.stack([x_centers, y_centers], axis=2)
    x0y0 = bboxes_centers - cell_size / 2 * inverse_scale
    x1y1 = bboxes_centers + cell_size / 2 * inverse_scale
    bboxes = np.concatenate([x0y0, x1y1], axis=2)
    return bboxes


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

def process_box(boxes, labels, img_size, class_num, anchors, img):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    params:
        boxes: [N, 5] shape, float32 dtype. `x_min, y_min, x_max, y_mix, mixup_weight`.
        labels: [N] shape, int64 dtype.
        class_num: int64 num.
        anchors: [9, 4] shape, float32 dtype.
    '''
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight.
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)
    # print('*'*10)
    # print(y_true_13.shape, y_true_26.shape, y_true_52.shape)

    # mix up weight default to 1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)
    # [N]
    best_match_idx = np.argmax(iou, axis=1)
    if best_match_idx.shape[0] == 0:
        print('error!!!')
    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    # cnt = 0
    grid_size_lst = [y_true_13.shape[0], y_true_26.shape[0], y_true_52.shape[0]]
    v3_anchor_lst = gen_v3_anchor_lst(anchors, grid_size_lst)
    N = boxes.shape[0]
    for i in range(N):
        num_match = 0

        idx = best_match_idx[i]
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]

        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

        for feature_map_group in range(3):
            ratio = ratio_dict[float(feature_map_group+1.)]
            for k in range(3):
                # print('add start...')
                anch_lay = v3_anchor_lst[feature_map_group][k]
                grid_size = y_true[feature_map_group].shape[0]
                # print('map_idx: %d, k: %d, grid_size: %d' %(feature_map_group, k, grid_size))
                # anch_lay = gen_anchor_lay(anchors, feature_map_group, k, grid_size)
                anch_lay = np.reshape(anch_lay, [-1, 4])
                # anch_lay *= ratio
                box = boxes[i, :]
                # print('box: ', box)

                ovr = IoU(box, anch_lay)
                # print('ovr.max', np.max(ovr))
                ovr = np.reshape(ovr, [grid_size, grid_size])
                # ovr_mask = ovr > 0.6
                POS_IOU_THRESH = 0.5
                NEG_IOU_THRESH = 0.1
                max_ovr = np.max(ovr)
                if max_ovr < POS_IOU_THRESH:
                    continue

                p_coor = np.where(ovr > POS_IOU_THRESH)
                # print('p_coor: ', p_coor)
                y_coor, x_coor = p_coor
                for y, x in zip(y_coor, x_coor):
                    if ovr[y, x] > y_true[feature_map_group][y, x, k, 4]:
                        # # print('add')
                        y_true[feature_map_group][y, x, k, :2] = box_centers[i]

                        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
                        y_true[feature_map_group][y, x, k, 4] = ovr[y, x]
                        y_true[feature_map_group][y, x, k, 5 + c] = 1.
                        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

                        num_match += 1

                        if DEBUG:
                            anchor_i = y * grid_size + x
                            anch_box = anch_lay[anchor_i]
                            anch_box = list(anch_box)
                            anch_box = list(map(int, anch_box))
                            print(anch_box)
                            img = cv2.rectangle(img, (anch_box[0], anch_box[1]), (anch_box[2], anch_box[3]), (0, 0, 255))

                ovr_vs_iou = np.less(ovr, NEG_IOU_THRESH)
                ovr_vs_y_true = np.less(y_true[feature_map_group][:, :, k, 4], NEG_IOU_THRESH)
                cond = np.logical_and(ovr_vs_iou, ovr_vs_y_true)
                minus_one = np.ones_like(ovr) * -1.
                y_true[feature_map_group][:, :, k, 4] = np.where(cond, minus_one, y_true[feature_map_group][:, :, k, 4])

        if DEBUG:
            box = boxes[i, :]
            box = list(box)
            box = list(map(int, box))
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))
            cv2.imwrite('./test_anch_match.jpg', img)

        # if num_match == 0:
        #     idx = best_match_idx[i]
        #     # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        #     feature_map_group = 2 - idx // 3
        #     # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        #     ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        #     x = int(np.floor(box_centers[i, 0] / ratio))
        #     y = int(np.floor(box_centers[i, 1] / ratio))
        #     k = anchors_mask[feature_map_group].index(idx)
        #     c = labels[i]
        #
        #     y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        #
        #     y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        #     y_true[feature_map_group][y, x, k, 4] = 1.
        #     y_true[feature_map_group][y, x, k, 5 + c] = 1.
        #     y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode):
    '''
    param:
        line: a line from the training/test txt file
        class_num: totol class nums.
        img_size: the size of image to be resized to. [width, height] format.
        anchors: anchors.
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    '''
    if not isinstance(line, list):
        img_idx, pic_path, boxes, labels = parse_line(line)
        img = cv2.imread(pic_path)
        # expand the 2nd dimension, mix up weight default to 1.
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
    else:
        # the mix up case
        _, pic_path1, boxes1, labels1 = parse_line(line[0])
        img1 = cv2.imread(pic_path1)
        img_idx, pic_path2, boxes2, labels2 = parse_line(line[1])
        img2 = cv2.imread(pic_path2)

        img, boxes = mix_up(img1, img2, boxes1, boxes2)
        labels = np.concatenate((labels1, labels2))

    if boxes.shape[0] == 0:
        print('tp1 error! boxes.shape: ', boxes.shape)
    if mode == 'train':
        # random color jittering
        # NOTE: applying color distort may lead to bad performance sometimes
        # img = random_color_distort(img)
        pic_path_lst = pic_path.split('/')
        assert (pic_path_lst[1] in ['data', 'data2']), 'pic_path_lst is error.'
        if pic_path_lst[3] == 'GoogleOpenImages':
            img, boxes = random_expand(img, boxes, 5)
        if boxes.shape[0] == 0:
            print('tp2 error! boxes.shape: ', boxes.shape)
        # random expansion with prob 0.5
        elif np.random.uniform(0, 1) > 0.5:
            img, boxes = random_expand(img, boxes, 2)


        # random cropping
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp)

        # random horizontal flip
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    else:
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1)
    img_bgr = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.
    # print('img.shape: ', img.shape, 'img_size: ', img_size)
    if boxes.shape[0] == 0:
        print('tp error! boxes.shape: ', boxes.shape)
    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors, img_bgr)
    return img_idx, img, y_true_13, y_true_26, y_true_52


def get_batch_data(batch_line, class_num, img_size, anchors, mode, multi_scale=False, mix_up=False, interval=10):
    '''
    generate a batch of imgs and labels
    param:
        batch_line: a batch of lines from train/val.txt files
        class_num: num of total classes.
        img_size: the image size to be resized to. format: [width, height].
        anchors: anchors. shape: [9, 2].
        mode: 'train' or 'val'. if set to 'train', data augmentation will be applied.
        multi_scale: whether to use multi_scale training, img_size varies from [320, 320] to [640, 640] by default. Note that it will take effect only when mode is set to 'train'.
        interval: change the scale of image every interval batches. Note that it's indeterministic because of the multi threading.
    '''
    global iter_cnt
    mode = mode.decode()
    # multi_scale training
    if multi_scale and mode == 'train':
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
        img_size = random.sample(random_img_size, 1)[0]
    iter_cnt += 1

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

    # mix up strategy
    if mix_up and mode == 'train':
        mix_lines = []
        batch_line = batch_line.tolist()
        for idx, line in enumerate(batch_line):
            if np.random.uniform(0, 1) < 0.5:
                mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
            else:
                mix_lines.append(line)
        batch_line = mix_lines

    for line in batch_line:
        img_idx, img, y_true_13, y_true_26, y_true_52 = parse_data(line, class_num, img_size, anchors, mode)

        img_idx_batch.append(img_idx)
        img_batch.append(img)
        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)

    return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch


if __name__ == '__main__':
    import args
    anchors = args.anchors
    # anchors = np.reshape(anchors, [9, 2])
    # v3_anch = gen_v3_anchor_lst(anchors)
    # print(v3_anch[0][0].shape)
    # map_idx = 2
    # k = 0
    # grid_size = 13
    # anch_lay = gen_anchor_lay(anchors, map_idx, k, grid_size)
    # print(anch_lay)

    f = open(args.val_file, 'r')
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)
    batch_line = f.readlines(1)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, \
    y_true_52_batch = get_batch_data(batch_line, 4, args.img_size, anchors, b'train',
                                     multi_scale=False, mix_up=False, interval=10)
    print('Done.')