import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

with open('wh.txt','r') as f:
    boxes = f.readlines()

ws = []
hs = []

for b in boxes:
    strList = b.split(' ')
    ws.append(float(strList[0]))
    hs.append(float(strList[1]))

ws_np = np.array(ws)
hs_np = np.array(hs)

plt.scatter(ws_np, hs_np)
plt.show()

WH = np.array(list(zip(ws_np, hs_np))).reshape((len(ws), 2))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b','p']
k = 9

kmeans = KMeans(k)
kmeans.fit(WH)

print (kmeans.cluster_centers_)


