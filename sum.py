import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn import metrics

with open('data/train_annotations.txt', 'r') as f:
    anno_list = f.readlines()

print(len(anno_list))

ws = []
hs = []
anno_data = []
for anno_str in anno_list:
    anno_str = anno_str[:-1]
    tmp_list = anno_str.split(' ')
    tmp = {}
    tmp['image'] = tmp_list[0]


    img = cv2.imread(os.path.join('data/train', tmp['image']))
    height, width = img.shape[:2]

    print (height, width)

    height_r = height / 416.0
    width_r = width / 416.0

    tmp['objects'] = []

    block = []
    for i in range(1, len(tmp_list)):
        if (i % 5 == 1):
            continue
        block.append( int(tmp_list[i]) )
        if (i % 5 == 0):
            block.append(int(tmp_list[i-4]) - 1)
            tmp['objects'].append(block)

            ws.append(float(block[2]) / width_r)
            hs.append(float(block[3]) / height_r)
            block = []

    anno_data.append(tmp)


ws_np = np.array(ws) / 32
hs_np = np.array(hs) / 32

WH = np.concatenate((ws_np, hs_np), axis = 0).reshape((len(ws), 2))

k = 5

kmeans = KMeans(k)
kmeans.fit(WH)

print (kmeans.cluster_centers_)


