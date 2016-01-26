#!/usr/bin/env python
"""
Created on Fri Mar 21 16:41:51 2014

@author: sinkpoint
"""
import vtk
import numpy as np
import numpy.linalg as la
import sys

QB_DIST = 18
QB_NPOINTS = 50
SCALAR_NAME = 'FA'

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

print 'tracks:',len(tract_ids)
verts = vtk_to_numpy(polydata.GetPoints().GetData())
print 'verts:',len(verts)

pointdata = polydata.GetPointData()
for si in range(pointdata.GetNumberOfArrays()):
    sname =  pointdata.GetArrayName(si)
    if sname==SCALAR_NAME:
        scalars = vtk_to_numpy(pointdata.GetArray(si))


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

bg_val = 0.6
fig = mlab.figure(bgcolor=(bg_val,bg_val,bg_val))
scene = mlab.gcf().scene
fig.scene.render_window.aa_frames = 4
mlab.draw()

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
qb = QuickBundles(streamlines, dist_thr=QB_DIST,pts=QB_NPOINTS)
# bundle_distance_mam

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


import scipy
import scikits.bootstrap as bootstrap
import scipy.stats as stats
import numpy as np


len_cent = len(centroids)
fig, axes = plt.subplots(len_cent, sharex=True, sharey=True)
pal = sns.color_palette("deep", len_cent)

plt.xlabel('Position Index')

def stats_per_group(x):
    res = {'median':[],'qtile':[],'ci':[]}
    medians = np.median(x)
    res['median'] = medians
    lower_quartile, upper_quartile = np.percentile(x, [25,75])
    res['qtile'] = (upper_quartile, lower_quartile)
#    res['ci'] = np.percentile(x, [2.5,97.5])
    iqr = upper_quartile - lower_quartile
    upper_whisker = x[x<=upper_quartile+1.5*iqr].max()
    lower_whisker = x[x>=lower_quartile-1.5*iqr].min()
    res['whisk'] = (lower_whisker, upper_whisker)
    res['err'] = (np.abs(lower_whisker-medians), np.abs(upper_whisker-medians))
    print res
    return res

for ci, cent in enumerate(centroids):
    print '---- centroid:'

    from scipy.cluster.vq import kmeans2
    ind = clusters[ci]['indices']
    # cent_streams = streamlines[ind]
    # cent_scalars = stream_scalars[ind]

    # apply each centriod to all the points
    # instead of only their centroid assignments
    cent_streams = streamlines
    cent_scalars = stream_scalars
    cent_verts = np.vstack(cent_streams)
    cent_scalars = np.concatenate(cent_scalars)
    c, labels = kmeans2(cent_verts, cent, iter=1)

    cid = np.ones(len(labels))
    d = {'Value':cent_scalars, 'position':labels}
    df = pd.DataFrame(data=d)
    grp = df.groupby('position', sort=True)
    #cent_stats = grp.FA.mean().as_matrix()
    cent_stats = grp.Value.apply(lambda x:stats_per_group(x))
    # cent_std = grp.FA.apply(lambda x:np.std(x)).as_matrix()
    # # bootstrap 68% CI, or 1 standard deviation
    #cent_ci = grp.Value.apply(lambda x: stats.norm.interval(0.95,loc=np.median(x),scale=np.std(x))).as_matrix()

    cent_stats = cent_stats.unstack()
    cent_median_scalar = cent_stats['median'].tolist()
    x = [i for i in grp.groups]
    print x

    print cent_stats['median'].tolist()

    mcolor = pal[ci]
    if type(axes) is np.ndarray:
        cur_axe = axes[ci]
    else:
        cur_axe = axes
    cur_axe.set_ylabel(SCALAR_NAME)

    #cur_axe.fill_between(x, [s[0] for s in cent_ci], [t[1] for t in cent_ci], alpha=0.3, color=mcolor)

    cur_axe.fill_between(x, [s[0] for s in cent_stats['whisk'].tolist()], 
        [t[1] for t in cent_stats['whisk'].tolist()], alpha=0.3, color=mcolor)

    cur_axe.fill_between(x, [s[0] for s in cent_stats['qtile'].tolist()], 
        [t[1] for t in cent_stats['qtile'].tolist()], alpha=0.3, color=mcolor)

    cur_axe.errorbar(x, cent_stats['median'].tolist(), yerr=[[s[0] for s in cent_stats['err'].tolist()], 
        [t[1] for t in cent_stats['err'].tolist()]], color=mcolor, alpha=0.5)    

    cur_axe.scatter(x,cent_stats['median'].tolist(), c=mcolor)    


# scene.renderer.render_window.set(alpha_bit_planes=1,multi_samples=0)
# scene.renderer.set(use_depth_peeling=True,maximum_number_of_peels=4,occlusion_ratio=0.1)
    ran_colors = np.random.random_integers(255, size=(len(cent),4))
    ran_colors[:,-1] = 255
    mypts = mlab.points3d(cent_verts[:,0],cent_verts[:,1],cent_verts[:,2],labels, opacity=0.2, mode='2dvertex')
    #mlab.points3d(cent_verts[:,0],cent_verts[:,1],cent_verts[:,2],cent_scalars, opacity=0.6, mode='2dvertex')

    delta = len(cent) - len(cent_median_scalar)
    if delta > 0:
        cent_median_scalar = np.pad(cent_median_scalar, (0,delta), mode='constant', constant_values=0)
    print len(cent),'=?=',len(cent_median_scalar)
    mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], x, colormap='blue-red', tube_radius=0.1, opacity=1)
    mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], cent_median_scalar, tube_radius=0.6, opacity=0.5)
    mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], cent_median_scalar, color=mcolor, tube_radius=1.5, opacity=0.2)
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
#plt.switch_backend('Qt4Agg')
mg = plt.get_current_fig_manager()
mg.resize(*mg.window.maxsize())
plt.show(block=False)
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

