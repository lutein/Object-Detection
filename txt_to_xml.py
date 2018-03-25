# txt_to_xml.py
# encoding:utf-8

from xml.dom.minidom import Document
import cv2
import os
import numpy as np
from math import cos, sin, atan, pi
def _mkanchors(x_ctr, y_ctr, ws, hs):
    anchors = np.array((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    #print anchors
    return anchors

def rotate(xy, cxcy, theta):
    return (
        cos(theta) * (xy[0] - cxcy[0]) - sin(theta) * (xy[1] - cxcy[1]) + cxcy[0],
        sin(theta) * (xy[0] - cxcy[0]) + cos(theta) * (xy[1] - cxcy[1]) + cxcy[1]
    )

def poly_to_box2d(poly):
    """
    polys: 1*8
    """
    assert (len(poly) == 8)
    cx = (poly[0] + poly[4])/2
    cy = (poly[1] + poly[5])/2
    delta_y = poly[3] - poly[1]
    delta_x = poly[2] - poly[0]
    angle = atan(delta_y/(delta_x+0.0000001))

    box2d = np.zeros(4)
    x0, y0 = rotate((poly[0], poly[1]), (cx, cy), -angle)
    x1, y1 = rotate((poly[4], poly[5]), (cx, cy), -angle)
    w0 = abs(x1 - x0)
    h0 = abs(y1 - y0)
    #box2d[...] = cx, cy, w0, h0, angle
    box2d = _mkanchors(cx, cy, w0, h0)

    return box2d

def find_rec(box):
    x0, x1 = min(box[0], box[2], box[4], box[6]), max(box[0], box[2], box[4], box[6])
    y0, y1 = min(box[1], box[3], box[5], box[7]), max(box[1], box[3], box[5], box[7])
    return np.array([x0, y0, x1, y1])

def parse_gt(gt_file):
    boxes = []
    ignore_label = []
    num = 0
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if line != "":
                parts= line.split(',')
                box = map(lambda x: float(x), parts[:8])
                box = find_rec(list(box))
                #print(box)
                text = ','.join(parts[8:])
                boxes.append(list(box))
                num += 1
                if text != '###':
                    ignore_label.append(1)
                else:
                    ignore_label.append(2)
    boxes = np.hstack((np.float32(boxes).reshape(num,4),np.float32(ignore_label).reshape(-1,1)))
    return boxes

def generate_xml(name,filename,img_size,class_ind):
    doc = Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name=name+'.jpg'

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The KITTI Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)
    boxes = parse_gt(filename)
    for box in boxes:
        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        if box[-1] == 1.0:
            title_text = doc.createTextNode('text')
        else:
            title_text = doc.createTextNode('not_text')
        title.appendChild(title_text)
        object.appendChild(title)

        title = doc.createElement('pose')
        title_text = doc.createTextNode('0')
        title.appendChild(title_text)
        object.appendChild(title)

        title = doc.createElement('truncated')
        title_text = doc.createTextNode('0')
        title.appendChild(title_text)
        object.appendChild(title)

        title = doc.createElement('difficult')
        title_text = doc.createTextNode('0')
        title.appendChild(title_text)
        object.appendChild(title)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(box[0])))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(box[1])))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(box[2])))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(box[3])))
        title.appendChild(title_text)
        bndbox.appendChild(title)


    f = open('Annotations/'+name+'.xml','w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()



if __name__ == '__main__':
    class_ind=('Car')
    cur_dir=os.getcwd() #current dir
    labels_dir=os.path.join(cur_dir,'gt')
    fn = open('name_list.txt','w')
    for parent, dirnames, filenames in os.walk(labels_dir):
        for file_name in filenames:
            full_path=os.path.join(parent, file_name)
            #f=open(full_path)
            #split_lines = f.readlines()
            name= file_name[:-4]
            img_name=name+'.jpg'
            img_path=os.path.join('./JPEGImages',img_name)
            if os.path.getsize(full_path)==0:
                os.remove(img_path)
            else:
                fn.writelines(name+'\n')
                img_size=cv2.imread(img_path).shape
                generate_xml(name,full_path ,img_size,class_ind)
        fn.close()
print('all txts has converted into xmls')
