#!/usr/bin/python
"""
Created on Fri Mar 21 16:41:51 2014

@author: sinkpoint
"""
import vtk
import numpy as np
import numpy.linalg as la


import sys
filename=sys.argv[1]

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(filename)
reader.Update()

polydata = reader.GetOutput()

from vtk.util.numpy_support import vtk_to_numpy

tract_ids = []
for i in range(polydata.GetNumberOfCells()):
    pids =  polydata.GetCell(i).GetPointIds()
    ids = [ pids.GetId(p) for p in range(pids.GetNumberOfIds())]
    tract_ids.append(ids)

print len(tract_ids)
verts = vtk_to_numpy(polydata.GetPoints().GetData())
print len(verts)
scalars = vtk_to_numpy(polydata.GetPointData().GetScalars())
print len(scalars)

print verts
print scalars

streamlines = []
stream_scalars = []
for i in tract_ids:
    streamlines.append(verts[i])
    stream_scalars.append(scalars[i])
streamlines = np.array(streamlines)
stream_scalars = np.array(stream_scalars)
# get total average direction (where majority point towards)
avg_d = np.zeros(3)
# for line in streams:
#     d = np.array(line[-1]) - np.array(line[0])
#     d = d / la.norm(d)
#     avg_d += d
#     avg_d /= la.norm(avg_d)   


avg_com = np.zeros(3)
avg_mid = np.zeros(3)






from dipy.segment.quickbundles import QuickBundles
from dipy.tracking import metrics as tm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

strl_len = [len(l) for l in streamlines]
stl_ori = np.array([np.abs(tm.mean_orientation(l)) for l in streamlines])


from mayavi import mlab

fig = mlab.figure()
scene = mlab.gcf().scene


#scipy.io.savemat("1.mat",{'streamlines':streamlines})
#need to transpose each stream array for AFQ in malab
# run with cmd
# @MATLAB: ts = cellfun(@transpose,streamlines,'UniformOutput',false)

#print streamlines




# sns.plt.figure()
# sns.distplot(strl_len, hist=False, rug=True)
# plt.show()

# from sklearn import mixture

# clf = mixture.GMM(n_components=#, n_iter=100, covariance_type='diag')
# clf.fit(strl_len)

# y=clf.predict(strl_len)


# fig = mlab.figure()
# scene = mlab.gcf().scene
# mlab.points3d(stl_ori[:,0],stl_ori[:,1],stl_ori[:,2],y, scale_factor=1, mode='2dvertex')
# #mlab.points3d(avg_mid[0],avg_mid[1],avg_mid[2],color=(0,0,1), mode='axes',scale_factor=5)
# mlab.show()

# d = pd.DataFrame(data={'len':strl_len, 'class':y})
# g=sns.FacetGrid(d, row='class', aspect=2)
# g.map(sns.kdeplot, 'len', )

# import numpy.ma as ma
# bundles = ma.masked_array(streamlines, y)
# print 'all:',len(streamlines)
# print 'grp1:',len(bundles.data)

# newlines = []
# for i, line in enumerate(streamlines):
#     if y[i] == 0:
#         newlines.append(line)


# streamlines = newlines

qb = QuickBundles(streamlines, dist_thr=20.,pts=20)

centroids = qb.centroids
clusters = qb.clusters()

avg_d = np.zeros(3)
avg_com = np.zeros(3)
avg_mid = np.zeros(3)

#unify centroid list orders to point in the same general direction
for i, line in enumerate(centroids):   
    print 'ori:',tm.mean_orientation(line) 
    d = np.array(line[-1]) - np.array(line[0])
    #print line[-1],line[0],d
    d = d / la.norm(d)
    if d.dot(avg_d) < 0:
        centroids[i] = line[::-1]
        line = centroids[i]
        d*=-1
        print 'reverse'
    avg_d += d



for ci, cent in enumerate(centroids):
    from scipy.cluster.vq import kmeans2
    ind = clusters[ci]['indices']
    cent_streams = streamlines[ind]
    cent_scalars = stream_scalars[ind]
    cent_verts = np.vstack(cent_streams)
    cent_scalars = np.concatenate(cent_scalars)
    c, labels = kmeans2(cent_verts, cent, iter=1)

    cid = np.ones(len(labels))
    d = {'FA':cent_scalars, 'position':labels}
    df = pd.DataFrame(data=d)
    cent_stats = df.groupby('position').mean().FA.as_matrix()
    sns.plt.figure()
    sns.tsplot(cent_stats)
    sns.plt.show(block=False)



# scene.renderer.render_window.set(alpha_bit_planes=1,multi_samples=0)
# scene.renderer.set(use_depth_peeling=True,maximum_number_of_peels=4,occlusion_ratio=0.1)
    ran_colors = np.random.random_integers(255, size=(len(cent),4))
    ran_colors[:,-1] = 255
    mypts = mlab.points3d(cent_verts[:,0],cent_verts[:,1],cent_verts[:,2],labels, opacity=0.5,  mode='2dvertex')
    #mypts = mlab.points3d(cent_verts[:,0],cent_verts[:,1],cent_verts[:,2],cent_scalars, opacity=0.6, mode='2dvertex')

    delta = len(cent) - len(cent_stats)
    if delta > 0:
        cent_stats = np.pad(cent_stats, (0,delta), mode='constant', constant_values=0)
    print len(cent),'=?=',len(cent_stats)
    mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], cent_stats, tube_radius=1)
    #myplot = mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], range(len(cent)), tube_radius=1, opacity=0.5)

    # print mypts.module_manager.scalar_lut_manager.lut.table.to_array()
    mypts.module_manager.scalar_lut_manager.lut.table = ran_colors
    mypts.module_manager.scalar_lut_manager.lut.number_of_colors = len(ran_colors)
    # myplot.module_manager.scalar_lut_manager.lut.table = ran_colors
    # myplot.module_manager.scalar_lut_manager.lut.number_of_colors = len(ran_colors)



    # draw original fibers
    # orig = streams[i]
    # mlab.plot3d(orig[:,0], orig[:,1], orig[:,2], color=(1,1,1), opacity=0.5)

# from dipy.segment.quickbundles import QuickBundles
# qb = QuickBundles(curves, dist_thr=5.,pts=100)
# print qb.centroids
# for c in qb.centroids:
#     print c
#     mlab.plot3d(c[:,0],c[:,1],c[:,2], color=(0,1,0),tube_radius=0.5 )

#mlab.points3d(verts[:,0],verts[:,1],verts[:,2],scalars, scale_factor=0.05, mode='2dvertex')
# verts = np.concatenate(newlines, axis=0)
# mlab.points3d(verts[:,0],verts[:,1],verts[:,2],scale_factor=0.05, mode='2dvertex')
#mlab.points3d(avg_mid[0],avg_mid[1],avg_mid[2],color=(0,0,1), mode='axes',scale_factor=5)
mlab.show()



#print npts

#ren = vtk.vtkRenderer()
#renwin = vtk.vtkRenderWindow()
#renwin.AddRenderer(ren)

#c1 = clusters[0]
#print c1['hidden']
# ren = fvtk.ren()
# cam = fvtk.camera(ren, pos=(0,0,-1), viewup=(0,1,0))

# fvtk.clear(ren)
# fvtk.add(ren, fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05)))
# #fvtk.add(ren, fvtk.add(ren, fvtk.dots(c1['hidden'], fvtk.colors.red, opacity=0.5, dot_size=1)))
# fvtk.add(ren, fvtk.add(ren, fvtk.streamtube(centroids, colormap, linewidth=0.4)))


# #mapper = vtk.vtkPolyDataMapper()
# #mapper.SetInput(reader.GetOutput())

# #actor=vtk.vtkActor()
# #actor.SetMapper(mapper)

# #ren.AddActor(actor)

# #iren = vtk.vtkRenderWindowInteractor()
# #iren.SetRenderWindow(renwin)
# #iren.Initialize()
# #iren.Start()

# fvtk.show(ren)

