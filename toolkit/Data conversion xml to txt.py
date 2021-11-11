import os
import sys
import xml.etree.ElementTree as ET
import glob

def xml2txt(xml, txt):

    os.chdir(xml)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')

    for i, file in enumerate(annotations):

        file_save = file.split('.')[0]+'.txt'
        file_txt = os.path.join(txt, file_save)
        f_w = open(file_txt, 'w')

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text

                xmlbox = obj.find('bndbox')
                xn = xmlbox.find('xmin').text
                xx = xmlbox.find('xmax').text
                yn = xmlbox.find('ymin').text
                yx = xmlbox.find('ymax').text
                #print xn
                f_w.write('pod' + ' ' + xn + ' ' + yn + ' ' + xx + ' ' + yx + '\n')


xml = '/media/zhengda/共享文件存档/处理好的pod/VOCdevkit-old/VOC2007/Annotations' #xml目录
txt = '/media/zhengda/共享文件存档/处理好的pod/VOCdevkit-old/VOC2007/txt' #txt目录

xml2txt(xml, txt)
