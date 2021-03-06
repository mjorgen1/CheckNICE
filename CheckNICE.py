import sys
sys.path.insert(0, '/home/mackenzie/GitRepo/NICE/cpp/build/interface')
import pylab as plt
from Nice4Py import KMean as N_KMean
import numpy as np
from sklearn.cluster import KMeans as S_KMeans
import time

#generate test data
#DATASET
numCenters = 5 #K?
numPoints = 2000
numTotalEle = numCenters*numPoints
dim = 2 #dimensions/features
dia = 100 #standard deviation

sideLength = 150 #range

cov = np.identity(dim) * dia

#creates a new array
data = np.empty([numCenters*numPoints, dim])
dataSize = numCenters*numPoints
print "The data size is: ", dataSize

for i in np.arange(numCenters):
  mean = np.random.uniform(0, sideLength, dim)
  data[i*numPoints, :] = mean
  points = np.random.multivariate_normal(mean, cov, numPoints-1)
  data[i*numPoints+1:(i+1)*numPoints, :] = points

#all data going to C++ needs to be typed
# np.float32
X = np.array(data, dtype=np.float32)

#plots the data created
plt.figure()
plt.scatter(x=data[:,0], y=data[:,1], c='gray', s = 10)
plt.title('Data Before Clustering')

#runs NICEkmeans and SKLearn
NICE_KMeans = N_KMean("cpu")
SKLearn_KMeans = S_KMeans(numCenters)

start = time.time()
NICE_KMeans.fit(X, numTotalEle, dim, numCenters)
end = time.time()
print ("Time to fit in NICE: ", (end-start))
start = time.time()
SKLearn_KMeans = SKLearn_KMeans.fit(data)
end = time.time()
print ("Time to fit in Sklearn: ", (end-start))

N_labels = np.zeros(numTotalEle, dtype=np.float32)
NICE_KMeans.getLabels(N_labels, numTotalEle, 1)
N_clusters = np.zeros((dim,numCenters), dtype=np.float32)
NICE_KMeans.getCenters(N_clusters, dim, numCenters)

S_labels = SKLearn_KMeans.labels_
S_clusters = SKLearn_KMeans.cluster_centers_

#plotting
LABEL_COLOR_MAP = {0: 'r',
                   1: 'b',
                   2: 'g',
                   3: 'c',
                   4: 'm',
                   5: 'y'
                   }
MARKER_MAP = {0: 'X',
              1: 'o',
              2: 's',
              3: '*',
              4: '^',
              5: 'p',
             }
#plot NICE
plt.figure()
label_color = [LABEL_COLOR_MAP[l] for l in N_labels]
plt.scatter(x=X[:,0], y=X[:,1], c=label_color, s=10)
plt.scatter(x=N_clusters[0,:], y=N_clusters[1,:], marker='x', s=100, c='black')
plt.title('NICE KMeans')
#plt.show()

#plot sklearn
plt.figure()
label_color2 = [LABEL_COLOR_MAP[s] for s in S_labels]
plt.scatter(x=data[:,0], y=data[:,1], c=label_color2, s = 10)
plt.scatter(x=S_clusters[:,0], y=S_clusters[:,1], marker='x', s=100, c='black')
plt.title("Sklearn KMeans")
plt.show()