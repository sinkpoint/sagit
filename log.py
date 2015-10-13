# IPython log file


import numpy as np
from vtkFileIO import vtkToStreamlines
streams = vtkToStreamlines('c50_test.vtk')
streams
streams.shape
len(streams)
np.concatenate(streams)
len(np.concatenate(streams))
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
p = StandardScaler().fit_transform(points)
points = np.concatenate(streams)
p = StandardScaler().fit_transform(points)
p
