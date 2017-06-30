#! /usr/bin/env python

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
#import austin's python lib into here

#DATASET
numCenters = 5
numPoints = 50
dim = 2
dia = 5

sideLength = 500
cov = np.identity(dim) * dia

#creates a new array
data = np.empty([numCenters*numPoints, dim])

for i in np.arange(numCenters):
  mean = np.random.uniform(0, sideLength, dim)
  data[i*numPoints, :] = mean
  points = np.random.multivariate_normal(mean, cov, numPoints-1)
  data[i*numPoints+1:(i+1)*numPoints, :] = points

#Kmeans/scatterplot
clf = KMeans(n_clusters=5)
clf.fit(data)

labels = clf.labels_
centers = clf.cluster_centers_

#plotting
LABEL_COLOR_MAP = {0: 'r',
                   1: 'b',
                   2: 'g',
                   3: 'c',
                   4: 'm',
                   5: 'y'
                   }


label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(x=data[:,0], y=data[:,1], c=label_color, s = 10)
plt.scatter(x=centers[:,0], y=centers[:,1], marker='X', s=75, c='black') #LABEL_COLOR_MAP.values()
plt.title("Sklearn Cluster")
plt.show()

#saving the random data set
#np.savetxt('data_k'+str(numCenters)+'_p'+str(numPoints)+'_d'+str(dim)+'_c'+str(dia)+'.txt', data, delimiter=',')
