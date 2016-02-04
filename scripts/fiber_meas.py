#!/usr/bin/env python
"""
Created on Fri Mar 21 16:41:51 2014

@author: sinkpoint
"""
import numpy as np
import numpy.linalg as la
import scipy
import sys
import vtk
#import scikits.bootstrap as bootstrap
import scipy.stats as stats

from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dipy.segment.quickbundles import QuickBundles
from dipy.tracking import metrics as tm
from mayavi import mlab
from scipy.cluster.vq import kmeans2

import os.path as path

QB_DIST = 18
QB_NPOINTS = 50
SCALAR_NAME = 'FA'
LOCAL_POINT_ASSIGN = False

filename=sys.argv[1]
filebase = path.basename(filename).split('.')[0]

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(filename)
reader.Update()

polydata = reader.GetOutput()


tract_ids = []
for i in range(polydata.GetNumberOfCells()):
    # get point ids in [[ids][ids...]...] format
    pids =  polydata.GetCell(i).GetPointIds()
    ids = [ pids.GetId(p) for p in range(pids.GetNumberOfIds())]
    tract_ids.append(ids) 
print 'tracks:',len(tract_ids)

verts = vtk_to_numpy(polydata.GetPoints().GetData())
print 'verts:',len(verts)

scalars = []
groups = []

pointdata = polydata.GetPointData()
for si in range(pointdata.GetNumberOfArrays()):
    sname =  pointdata.GetArrayName(si)
    if sname==SCALAR_NAME:
        scalars = vtk_to_numpy(pointdata.GetArray(si))
    if sname=='group':
        groups = vtk_to_numpy(pointdata.GetArray(si))


streamlines = []
stream_scalars = []
stream_groups = []
stream_pids = []
for i in tract_ids:
    # index np.array by a list will get all the respective indices
    streamlines.append(verts[i])
    stream_scalars.append(scalars[i])
    stream_groups.append(groups[i])
    stream_pids.append(i)

streamlines = np.array(streamlines)
stream_scalars = np.array(stream_scalars)
stream_groups = np.array(stream_groups)
stream_pids = np.array(stream_pids)

# get total average direction (where majority point towards)
avg_d = np.zeros(3)
# for line in streams:
#     d = np.array(line[-1]) - np.array(line[0])
#     d = d / la.norm(d)
#     avg_d += d
#     avg_d /= la.norm(avg_d)


avg_com = np.zeros(3)
avg_mid = np.zeros(3)






strl_len = [len(l) for l in streamlines]
stl_ori = np.array([np.abs(tm.mean_orientation(l)) for l in streamlines])


"""
    Use quickbundles to find centroids
"""
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
    ori = np.array(tm.mean_orientation(line))
    print 'ori:',ori
    #d = np.array(line[-1]) - np.array(line[0])
    #print line[-1],line[0],d
    # get the unit vector of the mean orientation
    if i==0:
        avg_d = ori

    #d = d / la.norm(d) 
    dotprod = ori.dot(avg_d) 
    print 'dotprod',dotprod
    if dotprod < 0:
        print 'reverse',dotprod      
        centroids[i] = line[::-1]
        line = centroids[i]
        ori*=-1
    avg_d += ori




# prepare mayavi 3d viz


bg_val = 0.6
fig = mlab.figure(bgcolor=(bg_val,bg_val,bg_val))
scene = mlab.gcf().scene
fig.scene.render_window.aa_frames = 4
mlab.draw()

# prepare the plt plot
len_cent = len(centroids)
fig, axes = plt.subplots(len_cent, sharex=True, sharey=True)
pal = sns.color_palette("deep", len_cent)

plt.xlabel('Position Index')


DATADF = None

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
    return res

for ci, cent in enumerate(centroids):
    print '---- centroid:'

    ind = clusters[ci]['indices']

    if LOCAL_POINT_ASSIGN:
        """
            apply centroid to only their point assignments
            through quickbundles
        """
        cent_streams = streamlines[ind]
        cent_scalars = stream_scalars[ind]
        cent_groups = stream_groups[ind]
        cent_pids = stream_pids[ind]
    else:
        # apply each centriod to all the points
        # instead of only their centroid assignments
        cent_streams = streamlines
        cent_scalars = stream_scalars
        cent_groups = stream_groups
        cent_pids = stream_pids


    cent_verts = np.vstack(cent_streams)
    cent_scalars = np.concatenate(cent_scalars)
    cent_groups = np.concatenate(cent_groups)
    cent_pids = np.concatenate(cent_pids)

    c, labels = kmeans2(cent_verts, cent, iter=1)

    cid = np.ones(len(labels))
    d = {'value':cent_scalars, 'position':labels, 'group':cent_groups, 'pid':cent_pids}

    df = pd.DataFrame(data=d)
    if DATADF is None:
        DATADF = df
    else:
        pd.concat([DATADF, df])

    """ 
        plot each group by their position 
    """

    UNIQ_GROUPS = df.group.unique()
    print '# UNIQ GROUPS',UNIQ_GROUPS

    for gi, GRP in enumerate(UNIQ_GROUPS):
        subgrp = df[df['group']==GRP]

        posGrp = subgrp.groupby('position', sort=True)
        #cent_stats = posGrp.FA.mean().as_matrix()
        cent_stats = posGrp.value.apply(lambda x:stats_per_group(x))
        # cent_std = posGrp.FA.apply(lambda x:np.std(x)).as_matrix()
        # # bootstrap 68% CI, or 1 standard deviation
        #cent_ci = posGrp.value.apply(lambda x: stats.norm.interval(0.95,loc=np.median(x),scale=np.std(x))).as_matrix()

        cent_stats = cent_stats.unstack()
        cent_median_scalar = cent_stats['median'].tolist()
        x = [i for i in posGrp.groups]
        # print x

        # print cent_stats['median'].tolist()

        mcolor = np.array(pal[ci])
        if gi>0:
            mcolor*= 1./(1+gi)

        mcolor = tuple(mcolor)

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
            [t[1] for t in cent_stats['err'].tolist()]], color=mcolor, alpha=0.25)    

        cur_axe.plot(x,cent_stats['median'].tolist(), c=mcolor)    

        cur_axe.scatter(x,cent_stats['median'].tolist(), c=mcolor)    


    """
        Plot 3D Viz 
    """

# scene.renderer.render_window.set(alpha_bit_planes=1,multi_samples=0)
# scene.renderer.set(use_depth_peeling=True,maximum_number_of_peels=4,occlusion_ratio=0.1)
    ran_colors = np.random.random_integers(255, size=(len(cent),4))
    ran_colors[:,-1] = 255
    mypts = mlab.points3d(cent_verts[:,0],cent_verts[:,1],cent_verts[:,2],labels, 
        opacity=0.1, 
        scale_mode='none',
        scale_factor=0.2,
        line_width=1,
        mode='point')

    # print mypts.module_manager.scalar_lut_manager.lut.table.to_array()
    mypts.module_manager.scalar_lut_manager.lut.table = ran_colors
    mypts.module_manager.scalar_lut_manager.lut.number_of_colors = len(ran_colors)



    delta = len(cent) - len(cent_median_scalar)
    if delta > 0:
        cent_median_scalar = np.pad(cent_median_scalar, (0,delta), mode='constant', constant_values=0)
    #print len(cent),'=?=',len(cent_median_scalar)

    # calculate the displacement vector for all pairs
    uvw =  cent - np.roll(cent,1, axis=0)
    uvw[0] *= 0 
    uvw = np.roll(uvw,-1,axis=0)
    arrow_plot = mlab.quiver3d(
        cent[:,0], cent[:,1], cent[:,2], 
        uvw[:,0], uvw[:,1], uvw[:,2], 
        scalars=cent_median_scalar,
        #scalars=[i for i in range(len(cent))],
        scale_factor=1,
        #color=mcolor,

        mode='arrow')

    gsource = arrow_plot.glyph.glyph_source.glyph_source
    
    # for name, thing in inspect.getmembers(gsource):
    #      print name
    
    arrow_plot.glyph.color_mode = 'color_by_scalar'
    #arrow_plot.glyph.scale_mode = 'scale_by_scalar'
    #arrow_plot.glyph.glyph.clamping = True
    #arrow_plot.glyph.glyph.scale_factor = 5
    #print arrow_plot.glyph.glyph.glyph_source
    gsource.tip_length=0.4
    gsource.shaft_radius=0.2
    gsource.tip_radius=0.3

    tube_plot = mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], cent_median_scalar, color=mcolor, tube_radius=0.2, opacity=0.25)
    tube_filter = tube_plot.parent.parent.filter
    tube_filter.vary_radius = 'vary_radius_by_scalar'
    tube_filter.radius_factor = 10

    #mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], cent_median_scalar, color=mcolor, tube_radius=1.5, opacity=0.2)
    #myplot = mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], range(len(cent)), tube_radius=1, opacity=0.5)

    # myplot.module_manager.scalar_lut_manager.lut.table = ran_colors
    # myplot.module_manager.scalar_lut_manager.lut.number_of_colors = len(ran_colors)


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


DATADF.to_csv('_'.join([filebase,SCALAR_NAME,'rawdata.csv']))

mg = plt.get_current_fig_manager()
mg.resize(*mg.window.maxsize())
plt.show(block=False)
mlab.show()


