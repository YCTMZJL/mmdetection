import numpy as np
import os
import glob
import math
from functools import cmp_to_key
import os
#from fire import Fire
from tqdm import tqdm
import cv2
import pickle as pkl
import traceback
import json
import emoji

def getAngleFromClockWise(p0, p1):
    vecX = p1[0] - p0[0]
    vecY = p1[1] - p0[1]
    Sidelen = (vecX**2 + vecY**2)**0.5
    angle = math.acos(vecY/Sidelen)
    if vecX < 0:
        angle = 2*math.pi - angle
    return angle


def transcoor(coor):
    center = np.mean(coor, axis=0)
    coor = list(coor)
    coor.sort(key=cmp_to_key(lambda b, a: getAngleFromClockWise(
        center, a) - getAngleFromClockWise(center, b)))
    coor = [coor[1], coor[2], coor[3], coor[0]]
    return np.array(coor)


def distance(p0, p1):
    x0 = float(p0[0])
    y0 = float(p0[1])
    x1 = float(p1[0])
    y1 = float(p1[1])
    res = ((x1-x0)**2 + (y1-y0)**2)**0.5
    return res


def is_vertical(poly):
    upper_left = poly[0]
    upper_right = poly[1]
    bottom_right = poly[2]
    bottom_left = poly[-1]

    width = abs(upper_left[0] - upper_right[0]) +\
        abs(bottom_left[0] - bottom_right[0])

    height = abs(upper_left[1] - bottom_left[1]) +\
        abs(upper_right[1] - bottom_right[1])
    return height / width > 1.5

def get_rect(contours_points):
    x, y, w, h = cv2.boundingRect(contours_points)
    return np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])

def gen_img_list(file_path='', save_path='test.txt', root_path='./'):
    with open(save_path,'w') as wr:
        for root, file_dir, filename in os.walk(file_path):
            for each_file_dir in file_dir:
                for sub_root, sub_dir, sub_filename in os.walk(os.path.join(root, each_file_dir)):
                    for each_sub_dir in sub_dir:
                        print(each_sub_dir)
                        for txt_line in glob.glob(os.path.join(sub_root, each_sub_dir)+'/*.txt'):
                            img_path = txt_line[:-4] + '.png'
                            if os.path.exists(img_path):
                                print(img_path)
                                root_name = root_path.split('/')[-1]
                                idx = img_path.find(root_name)
                                #idx = img_path.find('dataset')
                                #idx = img_path.find('dataset')
                                here_length = len(root_name) + 1
                                print(img_path[idx+here_length:])
                                wr.write(img_path[idx+here_length:])
                                wr.write('\n')

def gen_img_list_2(file_path='/mnt/storage01/Jerry-local/synth_text_chinese/dataset/filter_synth_1014', save_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/synthtext1014_filter_list.txt'):
    with open(save_path,'w') as wr:
        for root, file_dir, filename in os.walk(file_path):
            for each_file_dir in file_dir:
                print(each_file_dir)
                for sub_root, sub_dir, sub_filename in os.walk(os.path.join(root, each_file_dir)):
                    for each_sub_dir in sub_dir:
                        print(each_sub_dir)
                        for sub_sub_root, sub_sub_dir, sub_sub_filename in os.walk(os.path.join(sub_root, each_sub_dir)):
                            for each_sub_sub_dir in sub_sub_dir:
                                print(each_sub_sub_dir)
                                for txt_line in glob.glob(os.path.join(sub_sub_root, each_sub_sub_dir)+'/*.txt'):
                                    img_path = txt_line[:-4] + '.jpg'
                                    if os.path.exists(img_path):
                                        print(img_path)
                                        idx = img_path.find('dataset')
                                        print(img_path[idx+8:])
                                        wr.write(img_path[idx+8:])
                                        wr.write('\n')


def gen_img_gt(list_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/synthtext1014_filter_list.txt', root_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/train_images', save_root_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/train_gts'):
    with open(list_path, 'r') as list_file:
        for list_line in list_file:
            list_line = list_line.strip('\n')
            img_path = os.path.join(root_path, list_line)
            img_path = img_path.strip('\n')
            txt_path = img_path[:-4] + '.txt'
            if not os.path.exists(txt_path):
                print('not exist txt', txt_path)
                continue
            with open(txt_path, 'r') as txt_file:
                txt_lines =  txt_file.readlines()
                if len(txt_lines) == 0:
                    print('nothing in txt', txt_path)
                    continue
                txt_save_path = os.path.join(save_root_path, list_line) #dataset
                txt_save_path = txt_save_path + '.txt'
                print(txt_save_path)
                # check file exist
                txt_path_dir = txt_save_path[:txt_save_path.rfind('/')]
                print(txt_path_dir)
                if not os.path.exists(txt_path_dir):
                    os.makedirs(txt_path_dir)
                with open(txt_save_path, 'w') as wr:
                    for idx in range(int(txt_lines[0].strip('\n'))):
                        txt_line = txt_lines[idx+1]
                        txt_line = txt_line.strip('\n')
                        txt_line = txt_line.split(' ')
                        for idx in range(len(txt_line)):
                            wr.write(txt_line[(idx+1)%len(txt_line)])
                            if (idx+1) == len(txt_line): continue
                            wr.write(' ')
                        wr.write('\n') 
with open('./chinese_charset.dic', 'r') as rf:
    charset_dict = rf.readline()

def is_chinese(ch):    
    #uc=ch.decode('utf-8')     
    if '\u4e00' <= ch<='\u9fff':
        return True
    else:
        return False

def check_charset_frac(txt, f=0.7):
    chcnt = 0
    for ch in txt:
        if ch in charset_dict:
            chcnt += 1
    return float(chcnt)/(len(txt) + 0.0) > f      

def check_symb_frac(txt, f=0.7):
    """
    T/F return : T iff fraction of symbol/special-charcters in
                 txt is less than or equal to f (default=0.25).
    """
    chcnt=0
    line=txt#.decode('utf-8')
    for ch in line:
        if ch.isalnum() or is_chinese(ch):
            chcnt+=1
    return float(chcnt)/(len(txt)+0.0)>f
    #return np.sum([not ch.isalnum() for ch in txt])/(len(txt)+0.0) <= f

from string import punctuation

def is_good(txt, f=0.70):
    """
    T/F return : T iff the lines in txt (a list of txt lines)
                 are "valid".
                 A given line l is valid iff:
                     1. It is not empty.
                     2. symbol_fraction > f
                     3. Has at-least self.min_nchar characters
                     4. Not all characters are i,x,0,O,-
    """
    def is_txt(l):
        char_ex = punctuation #['i','I','o','O','0','-']
        chs = [ch in char_ex for ch in l]
        return not np.all(chs)

    return [ (len(l) >= 1 
             and check_symb_frac(l,f)
             and check_charset_frac(l,f)
             and is_txt(l)) for l in txt ]

def gen_crop_img_gt_json(list_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/synthtext1012_filter_list.txt', root_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/train_images', save_root_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/cropped_train_images_json_1012/', warp=True, flip=False):
    # save as pickle
    save_list_name = list_path.split('/')[-1]
    save_list_name = save_list_name[:save_list_name.rfind('.')]
    save_root_path = os.path.join(save_root_path, save_list_name)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    cnt = 0
    gts = []
    gt_json_path = os.path.join(save_root_path, 'gt.json')
    image_dir = os.path.join(save_root_path, 'images')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    with open(list_path, 'r') as list_file:
        for list_line in list_file:
            #if cnt > 2487610: #1550000:
            #    break
            #try:
            if True:
                list_line = list_line.strip('\n')
                img_path = os.path.join(root_path, list_line)
                img_path = img_path.strip('\n')
                txt_path = img_path[:-4] + '.txt'
                image = cv2.imread(img_path)
                with open(txt_path, 'r') as txt_file:
                    txt_lines =  txt_file.readlines()
                    #with open(pkl_save_path, 'wb') as wpkl:
                    if True:
                        # save only line
                        #data_dict_list = []
                        for idx in range(int(txt_lines[0].strip('\n'))):
                            #if (cnt == 576678):
                            #    print('err in cnt', cnt)
                            #    print(img_path)
                            #    cnt = cnt + 1
                            #    continue
                            #if cnt > 2487610: #1550000:
                            #    break
                            txt_line = txt_lines[idx+1]
                            txt_line = txt_line.strip('\n')
                            txt_line = txt_line.split(' ')
                            word = txt_line[0]
                            if not word == emoji.demojize(word):
                                print('---------------------with emoji', word)
                                continue
                            if not check_charset_frac(word, f=0.65):
                                #import pdb; pdb.set_trace()
                                print('--------------------not in charset', word)
                                continue
                            #if not is_good(word):
                            #    print('not good:', word)
                            #    continue
                            poly = np.array(txt_line[1:], dtype=np.float32)
                            poly = np.array(poly).reshape(4, 2).astype(np.int)
                            if not warp:
                                xmin = poly[:, 0].min()
                                xmax = poly[:, 0].max()
                                ymin = poly[:, 1].min()
                                ymax = poly[:, 1].max()

                                cropped_iamge = image[ymin:ymax+1, xmin:xmax+1, :]
                            else:
                                cur_poly = np.array(transcoor(poly), dtype=np.float32)
                                height1 = distance(cur_poly[0], cur_poly[3])
                                height2 = distance(cur_poly[1], cur_poly[2])
                                width1 = distance(cur_poly[0], cur_poly[1])
                                width2 = distance(cur_poly[2], cur_poly[3])
                                crop_height = int((height1+height2)/2)
                                crop_width = int((width1+width2)/2)
                                dst_poly = np.array(
                                    [(0, 0),
                                    (crop_width, 0),
                                    (crop_width, crop_height),
                                    (0, crop_height)], dtype=np.float32)
                                M = cv2.getPerspectiveTransform(cur_poly, dst_poly)
                                cropped_iamge = cv2.warpPerspective(
                                    image, M, (crop_width, crop_height))
                            height, width = cropped_iamge.shape[:2]

                            if height <= 2 or width <= 2:
                                continue
                            if flip and (1.5*width < height) and len(word) > 1:
                                cropped_iamge = np.flip(
                                    np.swapaxes(cropped_iamge, 0, 1), 0)
                            #cropped_data = cv2.imencode('.jpg', cropped_iamge)[
                            #    1].tostring()
 
                            #save_img
                            cv2.imwrite(os.path.join(image_dir, str(cnt) +'.jpg'), cropped_iamge)
                            gts.append(word)
                            print(cnt, word)
                            cnt += 1
                            #with open(gt_json_path, 'w') as wf:
                            #    json.dump(gts, wf)
 
            #except:
            #    print('*********err********', list_line)
            #    traceback.print_exc()
    with open(gt_json_path, 'w') as wf:
        json.dump(gts, wf)
 
def gen_crop_img_gt_json_mtwi(root_path='/mnt/storage01/Jerry-local/data/dataset/mtwi_2018_train/mtwi_2018_train', save_root_path='/mnt/storage01/Jerry-local/ocr-test/DB/datasets/mtwi_2018_train/cropped_train_images_json_mtwi1020_2/', warp=True, flip=True):
    txt_dir = 'txt_train'
    read_image_dir = 'image_train'
    cnt = 0
    gts = []
    gt_json_path = os.path.join(save_root_path, 'gt.json')
    image_dir = os.path.join(save_root_path, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for each_txt_path in glob.glob(os.path.join(root_path, txt_dir + '/*.txt')):
        print(each_txt_path)
        with open(each_txt_path, 'r') as txt_file:
            try:
            #if True:
                img_path = each_txt_path.split('/')[-1][:-4]+'.jpg'
                img_path = os.path.join(read_image_dir, img_path)
                img_path = os.path.join(root_path, img_path)
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                else:
                    import pdb; pdb.set_trace()
                if True:
                #with open(txt_path, 'r') as txt_file:
                    txt_lines =  txt_file.readlines()
                    #with open(pkl_save_path, 'wb') as wpkl:
                    if True:
                        # save only line
                        #data_dict_list = []
                        for txt_line in txt_lines:
                            txt_line = txt_line.strip('\n')
                            txt_line = txt_line.split(',')
                            word = txt_line[-1]
                            if '###' in word:
                                print('***All unknow***', word)
                                continue
                            poly = np.array(txt_line[:-1], dtype=np.float32)
                            poly = np.array(poly).reshape(4, 2).astype(np.int)
                            if not warp:
                                xmin = poly[:, 0].min()
                                xmax = poly[:, 0].max()
                                ymin = poly[:, 1].min()
                                ymax = poly[:, 1].max()

                                cropped_iamge = image[ymin:ymax+1, xmin:xmax+1, :]
                            else:
                                cur_poly = np.array(transcoor(poly), dtype=np.float32)
                                height1 = distance(cur_poly[0], cur_poly[3])
                                height2 = distance(cur_poly[1], cur_poly[2])
                                width1 = distance(cur_poly[0], cur_poly[1])
                                width2 = distance(cur_poly[2], cur_poly[3])
                                crop_height = int((height1+height2)/2)
                                crop_width = int((width1+width2)/2)
                                dst_poly = np.array(
                                    [(0, 0),
                                    (crop_width, 0),
                                    (crop_width, crop_height),
                                    (0, crop_height)], dtype=np.float32)
                                M = cv2.getPerspectiveTransform(cur_poly, dst_poly)
                                cropped_iamge = cv2.warpPerspective(
                                    image, M, (crop_width, crop_height))
                            height, width = cropped_iamge.shape[:2]

                            if height <= 2 or width <= 2:
                                continue
                            if flip and (1.5*width < height) and len(word) > 1:
                                print('~~~~~ enter flip ~~~~~~~')
                                cropped_iamge = np.flip(
                                    np.swapaxes(cropped_iamge, 0, 1), 0)
                            #cropped_data = cv2.imencode('.jpg', cropped_iamge)[
                            #    1].tostring()
 
                            #save_img
                            cv2.imwrite(os.path.join(image_dir, str(cnt) +'.jpg'), cropped_iamge)
                            gts.append(word)
                            print(cnt, word)
                            cnt += 1
                            with open(gt_json_path, 'w') as wf:
                                json.dump(gts, wf)
 
            except:
                print('*********err********', each_txt_path)
                import pdb;pdb.set_trace()
                traceback.print_exc()
    with open(gt_json_path, 'w') as wf:
        json.dump(gts, wf)
 
def process(list_path, file_path, root_path, save_root_path_db, save_root_path_crop):
    print('----gen img list----')
    #gen_img_list(file_path=file_path, save_path=list_path, root_path=root_path)
    print('----create db gts----')
    #gen_img_gt(list_path=list_path, root_path=root_path, save_root_path=save_root_path_db)
    print('----gen crop dataset----')
    gen_crop_img_gt_json(list_path=list_path, root_path=root_path, save_root_path=save_root_path_crop)

if __name__ == '__main__':
    #list_path = '/mnt/storage01/Jerry-local/ocr-test/DB/datasets/synthtext_datset/synthtext1012_filter_train_list.txt'
    #gen_img_list()
    #gen_img_gt()
    #gen_crop_img_gt_json()
    #gen_crop_img_gt_json_mtwi()
    list_path = '/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/0601_train_mix_list.txt'
    file_path = '/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/0601_train_mix'
    root_path = '/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/0601_train_mix' #'/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/0601_train_mix/train_dataset' #'/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/0601_train_mix'
    save_root_path_db = '/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/0601_train_mix' #'/mnt/storage01/Jerry-local/ocr-test/synth_text_chinese/dataset/filter_synth_simple_0419' #'/home/user/Jerry-local/data/dataset/synth_gy_from_hy'#'/mnt/storage01/Jerry-local/data/dataset/labelled_dataset/test_gts'#'/home/user/Jerry-local/data/dataset/synthtext_datset/train_gts'
    save_root_path_crop = '/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0525_hy/cropped_0601_train_mix' #'/home/user/Jerry-local/data/dataset/synthtext_datset/train_images/crop_filter_synth_simple_0514' #'/home/user/Jerry-local/data/dataset/synth_gy_from_hy/0419_mix/crop_0419_mix' #'/mnt/storage01/Jerry-local/ocr-test/synth_text_chinese/dataset/crop_filter_synth_simple_0419' #'/home/user/Jerry-local/data/dataset/synth_gy_from_hy/cropped_0414_jian' #'/mnt/storage01/Jerry-local/data/dataset/labelled_dataset/labelled0331_withsub/cropped_0331_json'
    #'/home/user/Jerry-locl/data/dataset/synthtext_datset/cropped_filter_more_synth_fan_0302_json_2'
    process(list_path=list_path, file_path=file_path, root_path=root_path, save_root_path_db=save_root_path_db, save_root_path_crop=save_root_path_crop)
