import os
from random import shuffle
import glob
import os
import cv2
from core.dataset import Dataset

def gen_train_val():
    lines = []
    pattern = '/data/Fire/fire-detection/filter_anno/fire_smoke-0528/*/*/*'
    image_list = glob.glob(pattern)
    for image in image_list:
        with open(image) as f:
            line = f.readlines()
        lst = line[0].strip().split(' ')
        if 'longmao' in image:
            lines.append(line[0])
        else:

            line_str = lst[1]
            sub_list = lst[2:]
            for i in range(int(len(sub_list)/5)):
                box = sub_list[i*5+1]+','+sub_list[i*5+2]+','+sub_list[i*5+3]+','+sub_list[i*5+4]+','+sub_list[i*5+0]
                line_str = line_str+' '+box
            lines.append(line_str+'\n')



    shuffle(lines)
    image_num=len(lines)
    # train_ratio = 0.7
    train_list = lines[0:int(len(lines)*0.7)]
    val_list = lines[int(len(lines)*0.7):]
    # neg_data = '/data/Fire/fire-detection/data/neg_data/*'
    # # video_pattern = '/data/Fire/fire-detection/neg/*'
    # vid_lists = glob.glob(neg_data)
    #
    # for lst in vid_lists:
    #     train_list.append(lst + '\n')
    #     image_num+=1
    # shuffle(train_list)
    with open('/home/amax/workspace-fire/tensorflow-yolov3/data/my_data/fire_train_add_longmao_new.txt','w') as f:
        for line in train_list:
            image_path = line.strip().split(' ')[0]
            if not os.path.exists(image_path):
                for type in ['.JPG', '.png']:
                    new_name = image_path.replace('.jpg',type)
                    if os.path.exists(new_name):
                        new_line = line.replace('.jpg', type)
                        f.write(new_line)
                        break
            else:
                f.write(line)
    with open('/home/amax/workspace-fire/tensorflow-yolov3/data/my_data/fire_val_add_longmao_new.txt','w') as f:
        for line in val_list:
            image_path = line.strip().split(' ')[0]
            if not os.path.exists(image_path):
                for type in ['.JPG', '.png']:
                    new_name = image_path.replace('.jpg', type)
                    if os.path.exists(new_name):
                        new_line = line.replace('.jpg', type)
                        f.write(new_line)
                        break
            else:
                f.write(line)


def image_exist(data_path):

    with open(data_path) as f:
        lines = f.readlines()
    for i,line in enumerate(lines):
        # print(line)
        image_path = line.strip().split(' ')[0]
        if not os.path.exists(image_path):
            print(image_path)
        # image = cv2.imread(image_path)
        # if image is None:
        #     print(image_path)
        # # print(line.split(' ')[1])

def get_data():
    dataset = Dataset('train')
    for data in dataset:
        print(len(data))

if __name__ == '__main__':
    gen_train_val()
    data_path = '/home/amax/workspace-fire/tensorflow-yolov3/data/my_data/fire_train_add_longmao.txt'
    image_exist(data_path)
    data_path = '/home/amax/workspace-fire/tensorflow-yolov3/data/my_data/fire_val_add_longmao.txt'
    image_exist(data_path)
    get_data()