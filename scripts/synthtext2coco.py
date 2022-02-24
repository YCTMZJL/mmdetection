import os.path as osp
import mmcv
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

def gen_crop_img_2_coco(list_path=None, root_path=None, save_root_path=None, warp=False, flip=False):
    # save as pickle
    save_list_name = list_path.split('/')[-1]
    save_list_name = save_list_name[:save_list_name.rfind('.')]
    save_root_path = os.path.join(save_root_path, save_list_name)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    cnt = 0
    gts = []
    gt_json_path = os.path.join(save_root_path, 'annotations_coco_format.json')
    image_dir = os.path.join(save_root_path, 'crops_images')
    
    #for coco format
    annotations = []
    images =[]
    object_count = 0


    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    with open(list_path, 'r') as list_file:
        for list_line in list_file:
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
                        total_word_lines = int(txt_lines[0].strip('\n'))
                        for idx in range(total_word_lines):
                            txt_line = txt_lines[idx+1]
                            txt_line = txt_line.strip('\n')
                            txt_line = txt_line.split(' ')
                            word = txt_line[0]
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
 
                            #save_img
                            cv2.imwrite(os.path.join(image_dir, str(cnt) +'.jpg'), cropped_iamge)
                            #gts.append(word)
                            image_id = cnt
                            cnt += 1
                            images.append(dict(
                                id = image_id,
                                file_name = '',
                                height = cropped_iamge.shape[0],
                                width = cropped_iamge.shape[1]
                                ))
                            print(cnt, word)

                            #for char
                            obj_cnt = 0
                            for char_idx in range(len(word)):
                                cur_char_idx = total_word_lines + 2 + cur_char_idx
                                char_line = txt_line[cur_char_idx]
                                char_line = char_line.strip('\n').split(' ')
                                char_label = char_line[0]
                                char_poly = np.float32(char_line[1:])
                                char_poly = char_poly.reshape(4,2).dtype(np.int) 
                                if warp:
                                    pass
                                else:
                                    char_poly = char_poly - np.array([xmin, ymin]) 
                                    char_xmin = char_poly[:, 0].min()
                                    char_xmax = char_poly[:, 0].max()
                                    char_ymin = char_poly[:, 1].min()
                                    char_ymax = char_poly[:, 1].max()
                                data_anno = dict(
                                        image_id = image_id,
                                        id=obj_cnt,
                                        bbox = [char_xmin, char_ymin, char_xmax - char_xmin, char_ymax - char_ymin],
                                        area=(char_ymax - char_ymin)*(char_xmax - char_xmin),
                                        segmentation = [char_poly],
                                        iscrowd=0)
                                annotations.append(data_anno)
                                obj_cnt += 1
                            #with open(gt_json_path, 'w') as wf:
                            #    json.dump(gts, wf)
 
            #except:
            #    print('*********err********', list_line)
            #    traceback.print_exc()
    coco_format_json = dict(
            images = images,
            annotations = annotations,
            category_id=[{'id':0, 'name':'character'}])
    with open(gt_json_path, 'w') as wf:
        json.dump(coco_format_json, wf)
 
def convert_synthtext_to_coco(ann_file, out_file, image_prefix):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        bboxes = []
        labels = []
        masks = []
        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))


            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'character'}])
    mmcv.dump(coco_format_json, out_file)

if __name__ = '__main__':
    list_path = 
    root_path = 
    save_root_path =
    gen_crop_img_2_coco(list_path=list_path, root_path=root_path, save_root_path=save_root_path)
