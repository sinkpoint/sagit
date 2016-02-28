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
import math
#import scikits.bootstrap as bootstrap
import scipy.stats as stats

from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.colors as colors
import matplotlib.cm as mcmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from optparse import OptionParser
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking import metrics as tm
from mayavi import mlab
from scipy.cluster.vq import kmeans2
import os.path as path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import stats

def position_stats(df):
        POS = df.position.unique()
        POS.sort()
        print len(POS)
        model = 'value ~ group'
        allpvals = None
        header = None
        DF = None
        for pos in POS:
            print pos
            data = df[df.position==pos]

            cross = smf.ols(model, data=data).fit()
            anova = sm.stats.anova_lm(cross, type=1)
            mcp = MultiComparison(data.value, data.group)
            rtp = mcp.allpairtest(stats.ttest_ind, method='bonferroni')
            #print rtp[0]
            mheader = [ "{}-{}".format(i[0],i[1]) for i in rtp[2] ]
            if not header or len(mheader) > len(header):
                header = mheader
            pvals = rtp[1][0][:,1]
            print mheader
            print pvals
            ndf = pd.DataFrame(data=[pvals], columns=mheader)
            if allpvals is None:
                allpvals = ndf
            else:
                allpvals = pd.concat([allpvals,ndf])
      
        flatten = allpvals.values.ravel()
        mcpres = multipletests(flatten, alpha=0.05, method='bonferroni')
        print mcpres
        corr_pvals = np.array(mcpres[1])
        print corr_pvals
        corr_pvals = np.reshape(corr_pvals, (len(POS),-1))
        print corr_pvals
        return pd.DataFrame(data=corr_pvals, columns=header)

def main():
    parser = OptionParser(usage="Usage: %prog [options] <tract.vtp>")
    parser.add_option("-d", "--dist", dest="dist", default=20, type='int', help="Quickbundle distance threshold")
    parser.add_option("-n", "--num", dest="num", default=50, type='int', help="Number of subdivisions along centroids")
    parser.add_option("-s", "--scalar", dest="scalar", default="FA", help="Scalar to measure")
    parser.add_option("--curvepoints", dest="curvepoints_file", help="Define a curve to use as centroid. Control points are defined in a csv file in the same space as the tract points. The curve is the vtk cardinal spline implementation, which is a catmull-rom spline.")
    parser.add_option("-l", "--local", dest="is_local", action="store_true", default=False, help="Measure from Quickbundle assigned streamlines. Default is to measure from all streamlines")
    (options, args) = parser.parse_args()

    QB_DIST = options.dist
    QB_NPOINTS = options.num
    SCALAR_NAME = options.scalar
    LOCAL_POINT_ASSIGN = options.is_local

    filename= args[0]
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
            groups = groups.astype(int)


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

    centroids = []
    if options.curvepoints_file: 
        LOCAL_POINT_ASSIGN = False
        cpoints = []
        ctrlpoints = np.loadtxt(options.curvepoints_file, delimiter=',')
        # have a separate vtkCardinalSpline interpreter for x,y,z
        curve = [vtk.vtkCardinalSpline() for i in range(3)]
        for c in curve:
            c.ClosedOff()

        for pi, point in enumerate(ctrlpoints):
            for i,val in enumerate(point):
                curve[i].AddPoint(pi,point[i])

        param_range = [0.0,0.0]
        curve[0].GetParametricRange(param_range)

        t = param_range[0]
        step = (param_range[1]-param_range[0])/(QB_NPOINTS-1.0)

        while t < param_range[1]:
            cp = [c.Evaluate(t) for c in curve]
            cpoints.append(cp)
            t = t + step

        centroids.append(cpoints)
        centroids = np.array(centroids)


    else:
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
    pal = sns.color_palette("bright", len_cent)


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

    """
        CENTROIDS
    """
    for ci, cent in enumerate(centroids):
        print '---- centroid:'

        if LOCAL_POINT_ASSIGN:
            """
                apply centroid to only their point assignments
                through quickbundles
            """
            ind = clusters[ci]['indices']
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
        cent_color = np.array(pal[ci])

        c, labels = kmeans2(cent_verts, cent, iter=1)

        cid = np.ones(len(labels))
        d = {'value':cent_scalars, 'position':labels, 'group':cent_groups, 'pid':cent_pids}

        df = pd.DataFrame(data=d)
        if DATADF is None:
            DATADF = df
        else:
            pd.concat([DATADF, df])

        """
            Perform stats
        """   

        pvalsDf = position_stats(df)
        logpvals = np.log(pvalsDf)*-1
        #print logpvals

        """ 
            plot each group by their position 
        """

        fig = plt.figure()
        ax1 = plt.subplot2grid((4,3),(0,0),colspan=3,rowspan=3)
        ax2 = plt.subplot2grid((4,3),(3,0),colspan=3,sharex=ax1)
        axes = [ax1,ax2]

        plt.xlabel('Position Index')     
        cent_patch = mpatches.Patch(color=cent_color, label='Centroid {}'.format(ci+1))
        cent_legend = axes[0].legend(handles=[cent_patch], loc=9)
        axes[0].add_artist(cent_legend)

        pvals = logpvals.mask(pvalsDf >= 0.05 ) 

        import matplotlib.ticker as mticker
        print pvals
        cmap = mcmap.Reds
        cmap.set_bad('w',1.)
        axes[1].pcolormesh(pvals.values.T,cmap=cmap,vmin=0, vmax=10, edgecolors='face', alpha=0.8)
        axes[1].yaxis.set_major_locator(mticker.MultipleLocator(base=1.0))
        axes[1].set_yticklabels(pvalsDf.columns.values.tolist(), minor=False)

        UNIQ_GROUPS = df.group.unique()
        UNIQ_GROUPS.sort()

        grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

        print '# UNIQ GROUPS',UNIQ_GROUPS

        legend_handles = []
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
            mcolor = np.array(grppal[gi])
            # if gi>0:
            #     mcolor*= 1./(1+gi)

            cent_color = tuple(cent_color)
            mcolor = tuple(mcolor)

            if type(axes) is list:
                cur_axe = axes[0]
            else:
                cur_axe = axes
            cur_axe.set_ylabel(SCALAR_NAME)
            # cur_axe.yaxis.label.set_color(cent_color)
            # cur_axe.tick_params(axis='y', colors=cent_color)

            #cur_axe.fill_between(x, [s[0] for s in cent_ci], [t[1] for t in cent_ci], alpha=0.3, color=mcolor)

            cur_axe.fill_between(x, [s[0] for s in cent_stats['whisk'].tolist()], 
                [t[1] for t in cent_stats['whisk'].tolist()], alpha=0.1, color=mcolor)

            cur_axe.fill_between(x, [s[0] for s in cent_stats['qtile'].tolist()], 
                [t[1] for t in cent_stats['qtile'].tolist()], alpha=0.4, color=mcolor)

            cur_axe.errorbar(x, cent_stats['median'].tolist(), yerr=[[s[0] for s in cent_stats['err'].tolist()], 
                [t[1] for t in cent_stats['err'].tolist()]], color=mcolor, alpha=0.1)    

            hnd, = cur_axe.plot(x,cent_stats['median'].tolist(), c=mcolor)   
            legend_handles.append(hnd) 

            # cur_axe.scatter(x,cent_stats['median'].tolist(), c=mcolor)   

        cur_axe.legend(legend_handles, UNIQ_GROUPS)


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

        # calculate the displacement vector for all pairs
        uvw =  cent - np.roll(cent,1, axis=0)
        uvw[0] *= 0 
        uvw = np.roll(uvw,-1,axis=0)
        arrow_plot = mlab.quiver3d(
            cent[:,0], cent[:,1], cent[:,2], 
            uvw[:,0], uvw[:,1], uvw[:,2], 
            scalars=cent_median_scalar,
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

        tube_plot = mlab.plot3d(cent[:,0], cent[:,1], cent[:,2], cent_median_scalar, color=cent_color, tube_radius=0.2, opacity=0.25)
        tube_filter = tube_plot.parent.parent.filter
        tube_filter.vary_radius = 'vary_radius_by_scalar'
        tube_filter.radius_factor = 10




    DATADF.to_csv('_'.join([filebase,SCALAR_NAME,'rawdata.csv']), index=False)

    mg = plt.get_current_fig_manager()
    # mg.resize(*mg.window.maxsize())
    plt.show(block=False)
    mlab.show()

if __name__ == '__main__':
    main()

