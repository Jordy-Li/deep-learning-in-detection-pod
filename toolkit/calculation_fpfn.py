import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

net_name = 'Faster R-CNN'

# 将annota_dir中的xml文件转成简单的txt文件保存在target_dir中
annota_dir = r'/media/zhengda/共享文件存档/tf-faster-rcnn-master/data/VOCdevkit2007/VOC2007/Annotations'
target_dir = r'/media/zhengda/共享文件存档/data_processing/result-txt-ori'

# 读取转换后的txt，与检测后生成的txt比较
label_path = r'/media/zhengda/共享文件存档/data_processing/result-txt-ori'
detect_path = r'/media/zhengda/共享文件存档/data_processing/result-txt'

def xml_to_txt(oriname):
    # 读取每个原图像的xml文件
    xml_file = os.path.join(annota_dir, oriname)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    f = open(os.path.join(target_dir, oriname[0:-3]+'txt'), 'w')
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = float(object.find('bndbox').find('xmin').text)
        Ymin = float(object.find('bndbox').find('ymin').text)
        Xmax = float(object.find('bndbox').find('xmax').text)
        Ymax = float(object.find('bndbox').find('ymax').text)
        # 左上右下位置xy
        f.write('%s %s %s %s %s\n' % (
            'pod', str(float(Xmin)), str(float(Ymin)), str(float(Xmax)), str(float(Ymax))))
    f.close()

def function(name):
    FP = 0
    FN = 0
    with open(os.path.join(detect_path, name)) as d_f:
        lines = d_f.readlines()
        pd = len(lines)
        for line in lines:
            count = 0
            line = line.replace('\n', '')
            line = line.split(' ')
            d_left = float(line[1])
            d_top = float(line[2])
            d_right = float(line[3])
            d_bottom = float(line[4])
            d_cover = (d_right - d_left) * (d_bottom - d_top)
            with open(os.path.join(label_path, name)) as g_f:
                lines = g_f.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    line = line.split(' ')
                    g_left = float(line[1])
                    g_top = float(line[2])
                    g_right = float(line[3])
                    g_bottom = float(line[4])
                    g_cover = (g_right - g_left) * (g_bottom - g_top)
                    if g_left > d_left:
                        left = g_left
                    else:
                        left = d_left
                    if g_right > d_right:
                        right = d_right
                    else:
                        right = g_right
                    if g_top > d_top:
                        top = g_top
                    else:
                        top = d_top
                    if g_bottom > d_bottom:
                        bottom = d_bottom
                    else:
                        bottom = g_bottom
                    if (right-left)<0 or (bottom-top)<0:
                        continue
                    inter_cover = (right - left) * (bottom - top)
                    uinon_cover = d_cover + g_cover - inter_cover
                    if inter_cover / uinon_cover >= 0.5:
                        count += 1
            if count == 0:
                FP += 1

    with open(os.path.join(label_path, name)) as g_f:
        lines = g_f.readlines()
        gt = len(lines)
        for line in lines:
            count = 0
            line = line.replace('\n', '')
            line = line.split(' ')
            g_left = float(line[1])
            g_top = float(line[2])
            g_right = float(line[3])
            g_bottom = float(line[4])
            g_cover = (g_right - g_left) * (g_bottom - g_top)
            with open(os.path.join(detect_path, name)) as d_f:
                lines = d_f.readlines()
                for line in lines:
                    line = line.replace('\n', '')
                    line = line.split(' ')
                    d_left = float(line[1])
                    d_top = float(line[2])
                    d_right = float(line[3])
                    d_bottom = float(line[4])
                    d_cover = (d_right - d_left) * (d_bottom - d_top)
                    if g_left > d_left:
                        left = g_left
                    else:
                        left = d_left
                    if g_right > d_right:
                        right = d_right
                    else:
                        right = g_right
                    if g_top > d_top:
                        top = g_top
                    else:
                        top = d_top
                    if g_bottom > d_bottom:
                        bottom = d_bottom
                    else:
                        bottom = g_bottom
                    if (right-left)<0 or (bottom-top)<0:
                        continue
                    inter_cover = (right - left) * (bottom - top)
                    uinon_cover = d_cover + g_cover - inter_cover
                    if inter_cover / uinon_cover >= 0.5:
                        count += 1
            if count == 0:
                FN += 1
    return gt, pd, FP, FN

def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae

def compute_mse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse

txt_list = os.listdir(label_path)
f_txt = open('./fnfp_vgg16_50001.txt', 'w')
f_txt.write('名称\t标注数目\t检测数目\tFP\tFN\n')
for name in txt_list:
    if name[-3:] == 'txt':
        gt, pd, fp, fn = function(name)
        f_txt.write('%s\t%s\t%s\t%s\t%s\n' % (
        name[0:-4], str(gt), str(pd), str(fp), str(fn)))
    else:
        print(name[-3:])
        continue
f_txt.close()










