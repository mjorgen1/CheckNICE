import sys
sys.path.insert(0, '/home/mackenzie/NICE/cpp/interface') #CHANGE

from Nice4Py import KMean
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import pylab as plt
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

#runs NICEkmeans
clf = KMean("cpu")

start = timer()
clf.fit(X, numTotalEle, dim, numCenters)
end = timer()
print ("Time to fit: ", (end-start))

labels = np.zeros(numTotalEle, dtype=np.float32)
clf.getLabels(labels,numTotalEle,1)
centers = np.zeros((dim,numCenters), dtype=np.float32)
clf.getCenters(centers,dim,numCenters)

#plotting
LABEL_COLOR_MAP = {0: 'r',
                   1: 'b',
                   2: 'g',
                   3: 'c',
                   4: 'm',
                   5: 'y'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(x=X[:,0], y=X[:,1], c=label_color)
plt.scatter(x=centers[0,:], y=centers[1,:], marker='x', s=100, c='black') #might need to fix this line
plt.title('NICE Kmeans')
plt.show()