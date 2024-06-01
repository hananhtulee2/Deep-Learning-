import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import numpy

img = plt.imread('1351402.png')
width = img.shape[0]
height = img.shape[1]

img1 = img  

img = img.reshape(width*height,3)

print(img.shape)
kmeans = KMeans(n_clusters=5,n_init=5).fit(img)

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

print(labels)

img2 = numpy.zeros_like(img)
print(img2)

for i in range(len(img2)):
    img2[i] = clusters[labels[i]]


img2 = img2.reshape(width,height,3)
plt.imshow(img2)
plt.show()
