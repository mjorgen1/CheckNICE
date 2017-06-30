import sys
sys.path.insert(0, '/home/mackenzie/NICE/cpp/interface') #CHANGE
import pylab as plt
from Nice4Py import KMean as N_KMean
import numpy as np
from sklearn.cluster import KMeans as S_KMeans
from timeit import default_timer as timer



#generate test data
#DATASET
numCenters = 5 #K?
numPoints = 50
numTotalEle = numCenters*numPoints
dim = 2 #dimensions/features
dia = 5 #standard deviation

sideLength = 500 #range

cov = np.identity(dim) * dia

#creates a new array
data = np.empty([numCenters*numPoints, dim])

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

start = timer()
NICE_KMeans.fit(X, numTotalEle, dim, numCenters)
end = timer()
print ("Time to fit: ", (end-start))
SKLearn_KMeans = SKLearn_KMeans.fit(data)

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

#plot NICE
plt.figure()
label_color = [LABEL_COLOR_MAP[l] for l in N_labels]
plt.scatter(x=X[:,0], y=X[:,1], c=label_color)
plt.scatter(x=N_clusters[0,:], y=N_clusters[1,:], marker='x', s=100, c='black') #might need to fix this line
plt.title('NICE KMeans')
#plt.show()

#plot sklearn
plt.figure()
label_color = [LABEL_COLOR_MAP[l] for l in S_labels]
plt.scatter(x=data[:,0], y=data[:,1], c=label_color, s = 10)
plt.scatter(x=S_clusters[:,0], y=S_clusters[:,1], marker='x', s=100, c='black') #LABEL_COLOR_MAP.values()
plt.title("Sklearn KMeans")
plt.show()