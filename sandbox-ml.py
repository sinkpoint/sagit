#!/usr/bin/python

#convert vtk to vtp files to filter
import sys
import numpy as np
from vtkFileIO import vtkToStreamlines

streams = vtkToStreamlines(sys.argv[1])
data = np.concatenate(streams)


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
# colors = np.hstack([colors] * 20)

# plt.scatter(norm_p[:, 0], norm_p[:, 1], zs=norm_p[:,2], color=colors[y_pred].tolist(), s=1)

# if hasattr(algo, 'cluster_centers_'):
#     centers = algo.cluster_centers_
#     center_colors = colors[:len(centers)]
#     plt.scatter(centers[:, 0], centers[:, 1], zs=centers[:,2], s=100, c=center_colors)
for line in streams:
    ax.plot(line[:,0], line[:,1], line[:,2])

plt.show()


import time

from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(data)

# Create a regular 3D grid with 50 points in each dimension
xmin, ymin, zmin = data.min(axis=0)
xmax, ymax, zmax = data.max(axis=0)
print xmin,xmax
print ymin,ymax
print zmin,zmax


xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]

# Evaluate the KDE on a regular grid...
coords = np.vstack([item.ravel() for item in [xi, yi, zi]]).T
print coords.shape
res = kde.score_samples(coords)
print res

density = res.reshape(xi.shape)

from mayavi import mlab

# Visualize the density estimate as isosurfaces
mlab.contour3d(xi, yi, zi, density, opacity=0.5)
mlab.axes()
mlab.show()



# from sklearn import cluster
# from sklearn.preprocessing import StandardScaler

# algo = cluster.DBSCAN()
# norm_p =StandardScaler().fit_transform(points)
# t0 = time.time()
# print 'start ',t0
# algo.fit(norm_p)
# t1 = time.time()
# print 'end ',t1
# print 'elapse %.2f' % (t1-t0)

# if hasattr(algo, 'labels_'):
#     y_pred = algo.labels_.astype(np.int)
# else:
#     y_pred = algo.predict(X)




