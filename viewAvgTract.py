#!/usr/bin/python
"""
Created on Fri Mar 21 16:41:51 2014

@author: sinkpoint
"""
import vtk
import numpy as np

from nibabel import trackvis as tv
from dipy.tracking import metrics as tm
from dipy.segment.quickbundles import QuickBundles
from dipy.viz import fvtk

import sys
filename=sys.argv[1]

reader = vtk.vtkPolyDataReader()
reader.SetFileName(filename)
reader.Update()

polydata = reader.GetOutput()

streamlines = []
for i in range(polydata.GetNumberOfCells()):
    pts = polydata.GetCell(i).GetPoints()
    npts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    streamlines.append(npts)

import scipy
scipy.io.savemat("1.mat",{'streamlines':streamlines})
#need to transpose each stream array for AFQ in malab
# run with cmd
# @MATLAB: ts = cellfun(@transpose,streamlines,'UniformOutput',false)

#print streamlines

qb = QuickBundles(streamlines, dist_thr=10.,pts=20)

centroids = qb.centroids
clusters = qb.clusters()
colormap = np.random.rand(len(centroids),3)


#print npts

#ren = vtk.vtkRenderer()
#renwin = vtk.vtkRenderWindow()
#renwin.AddRenderer(ren)

#c1 = clusters[0]
#print c1['hidden']
ren = fvtk.ren()
cam = fvtk.camera(ren, pos=(0,0,-1), viewup=(0,1,0))

fvtk.clear(ren)
fvtk.add(ren, fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05)))
#fvtk.add(ren, fvtk.add(ren, fvtk.dots(c1['hidden'], fvtk.colors.red, opacity=0.5, dot_size=1)))
fvtk.add(ren, fvtk.add(ren, fvtk.streamtube(centroids, colormap, linewidth=0.4)))


#mapper = vtk.vtkPolyDataMapper()
#mapper.SetInput(reader.GetOutput())

#actor=vtk.vtkActor()
#actor.SetMapper(mapper)

#ren.AddActor(actor)

#iren = vtk.vtkRenderWindowInteractor()
#iren.SetRenderWindow(renwin)
#iren.Initialize()
#iren.Start()

fvtk.show(ren)

