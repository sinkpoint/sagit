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


def getNiftiAsScalarField(filename):
    import nibabel as nib

    fimg = nib.load(filename)
    hdr = fimg.get_header()
    data = fimg.get_data()

    print data.shape

    af = fimg.get_affine()
    print af
    origin = af[:3,3]
    print origin
    
    stride = [af[0,0],af[1,1],af[2,2]]
    print stride

    if stride[0] < 0:
        print 'flipX'
        data = data[::-1,:,:]
        origin[0] *= -1
        #data = np.flipud(data)
    if stride[1] < 0:
        print 'flipY'           
        data = data[:,::-1,:]
        origin[1] *= -1
    if stride[2] < 0:
        print 'flipZ'           
        data = data[:,:,::-1]     
        origin[2] *= -1      

    src = mlab.pipeline.scalar_field(data)
    src.spacing = hdr.get_zooms()
    src.origin = origin
    print src.origin
    src.update_image_data = True    
    return src, data

def position_stats(df):
    from statsmodels.stats.weightstats import ztest
    from functools32 import partial, wraps
    POS = df.position.unique()
    POS.sort()
    print len(POS)
    model = 'value ~ group'
    allpvals = None
    header = None
    DF = None

    ttest_ind_nev = wraps(partial(stats.ttest_ind, equal_var=False))(stats.ttest_ind)
    mwu_test = wraps(partial(stats.mannwhitneyu, use_continuity=False))(stats.mannwhitneyu)

    print df

    stats_test = ttest_ind_nev
    for pos in POS:
        print pos
        data = df[df.position==pos]
        data = data.groupby(['sid']).mean()
        print data


        #cross = smf.ols(model, data=data).fit()
        #anova = sm.stats.anova_lm(cross, type=1)
        mcp = MultiComparison(data.value, data.group)

        rtp = mcp.allpairtest(stats_test, method='fdr_bh')
        #print rtp[0]
        mheader = [ "{}-{}".format(i[0],i[1]) for i in rtp[2] ]
        if not header or len(mheader) > len(header):
            header = mheader
        pvals = rtp[1][0][:,1]
        print pvals
        # print mheader
        # print pvals
        ndf = pd.DataFrame(data=[pvals], columns=mheader)
        if allpvals is None:
            allpvals = ndf
        else:
            allpvals = pd.concat([allpvals,ndf])
            
    corr_pvals = allpvals    
    # flatten = allpvals.values.ravel()
    # flatten = flatten * 2
    # mcpres = multipletests(flatten, alpha=0.05, method='fdr_bh')
    # print mcpres
    # corr_pvals = np.array(mcpres[1])
    # print corr_pvals
    # corr_pvals = np.reshape(corr_pvals, (len(POS),-1))

    print corr_pvals
    return pd.DataFrame(data=corr_pvals, columns=header)

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
        
def main():
    parser = OptionParser(usage="Usage: %prog [options] <tract.vtp>")
    parser.add_option("-d", "--dist", dest="dist", default=20, type='float', help="Quickbundle distance threshold")
    parser.add_option("-n", "--num", dest="num", default=50, type='int', help="Number of subdivisions along centroids")
    parser.add_option("-s", "--scalar", dest="scalar", default="FA", help="Scalar to measure")
    parser.add_option("--curvepoints", dest="curvepoints_file", help="Define a curve to use as centroid. Control points are defined in a csv file in the same space as the tract points. The curve is the vtk cardinal spline implementation, which is a catmull-rom spline.")
    parser.add_option("-l", "--local", dest="is_local", action="store_true", default=False, help="Measure from Quickbundle assigned streamlines. Default is to measure from all streamlines")
    parser.add_option('--reverse', dest='is_reverse', action='store_true', default=False, help='Reverse the centroid measure stepping order')
    parser.add_option('--background', dest='bg_file', help='Background NIFTI image')
    parser.add_option('--pairplot', dest='pairplot',)
    parser.add_option('--noviz',dest='is_viz', action='store_false', default=True)
    parser.add_option('--xrange', dest='xrange')
    parser.add_option('--yrange', dest='yrange')
    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(2)
        
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
    subjects = []
    pointdata = polydata.GetPointData()
    for si in range(pointdata.GetNumberOfArrays()):
        sname =  pointdata.GetArrayName(si)
        print sname
        if sname==SCALAR_NAME:
            scalars = vtk_to_numpy(pointdata.GetArray(si))
        if sname=='group':
            groups = vtk_to_numpy(pointdata.GetArray(si))
            groups = groups.astype(int)
        if sname=='tid':
            subjects = vtk_to_numpy(pointdata.GetArray(si))
            subjects = subjects.astype(int)


    streamlines = []
    stream_scalars = []
    stream_groups = []
    stream_pids = []
    stream_sids = []

    for i in tract_ids:
        # index np.array by a list will get all the respective indices
        streamlines.append(verts[i])
        stream_scalars.append(scalars[i])
        stream_groups.append(groups[i])
        stream_pids.append(i)
        stream_sids.append(subjects[i])

    streamlines = np.array(streamlines)
    stream_scalars = np.array(stream_scalars)
    stream_groups = np.array(stream_groups)
    stream_pids = np.array(stream_pids)
    stream_sids = np.array(stream_sids)

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

        if options.is_reverse:
            for i,c in enumerate(centroids):
                centroids[i] = c[::-1]


    # prepare mayavi 3d viz


    bg_val = 0.
    fig = mlab.figure(bgcolor=(bg_val,bg_val,bg_val))
    scene = mlab.gcf().scene
    fig.scene.render_window.aa_frames = 4
    mlab.draw()

    if options.bg_file:
        mrsrc, bgdata = getNiftiAsScalarField(options.bg_file)
        orie = 'z_axes'

        opacity=0.5
        slice_index = 0

        mlab.pipeline.image_plane_widget(mrsrc, opacity=opacity, plane_orientation=orie, slice_index=int(slice_index), colormap='black-white', line_width=0, reset_zoom=False)

    # prepare the plt plot
    len_cent = len(centroids)
    pal = sns.color_palette("bright", len_cent)


    DATADF = None

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
            cent_sids = stream_sids[ind]
        else:
            # apply each centriod to all the points
            # instead of only their centroid assignments
            cent_streams = streamlines
            cent_scalars = stream_scalars
            cent_groups = stream_groups
            cent_pids = stream_pids
            cent_sids = stream_sids


        cent_verts = np.vstack(cent_streams)
        cent_scalars = np.concatenate(cent_scalars)
        cent_groups = np.concatenate(cent_groups)
        cent_pids = np.concatenate(cent_pids)
        cent_sids = np.concatenate(cent_sids)
        cent_color = np.array(pal[ci])

        c, labels = kmeans2(cent_verts, cent, iter=1)

        cid = np.ones(len(labels))
        d = {'value':cent_scalars, 'position':labels, 'group':cent_groups, 'pid':cent_pids, 'sid':cent_sids}

        df = pd.DataFrame(data=d)
        if DATADF is None:
            DATADF = df
        else:
            pd.concat([DATADF, df])



        UNIQ_GROUPS = df.group.unique()
        UNIQ_GROUPS.sort()

        grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

        print '# UNIQ GROUPS',UNIQ_GROUPS


        """ 
            plot each group by their position 
        """

        fig = plt.figure(figsize=(14,7))
        ax1 = plt.subplot2grid((4,3),(0,0),colspan=3,rowspan=3)
        ax2 = plt.subplot2grid((4,3),(3,0),colspan=3,sharex=ax1)
        axes = [ax1,ax2]

        plt.xlabel('Position Index')     
        cent_patch = mpatches.Patch(color=cent_color, label='Centroid {}'.format(ci+1))
        cent_legend = axes[0].legend(handles=[cent_patch], loc=9)
        axes[0].add_artist(cent_legend)

        """
            Perform stats
        """   

        if len(UNIQ_GROUPS) > 1:
            pvalsDf = position_stats(df)
            logpvals = np.log(pvalsDf)*-1
            #print logpvals


            pvals = logpvals.mask(pvalsDf >= 0.05 ) 

            import matplotlib.ticker as mticker
            print pvals
            cmap = mcmap.Reds
            cmap.set_bad('w',1.)
            axes[1].pcolormesh(pvals.values.T,cmap=cmap,vmin=0, vmax=10, edgecolors='face', alpha=0.8)
            #axes[1].yaxis.set_major_locator(mticker.MultipleLocator(base=1.0))
            axes[1].set_yticks(np.arange(pvals.values.shape[1])+0.5, minor=False)
            axes[1].set_yticklabels(pvalsDf.columns.values.tolist(), minor=False)



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

            if options.xrange:
                plotrange = options.xrange.split(',')
                cur_axe.set_xlim([int(plotrange[0]), int(plotrange[1])])

            if options.yrange:
                plotrange = options.yrange.split(',')
                cur_axe.set_ylim([float(plotrange[0]), float(plotrange[1])])

        cur_axe.legend(legend_handles, UNIQ_GROUPS)


        """
            Plot 3D Viz 
        """

        scene.disable_render = True
        # scene.renderer.render_window.set(alpha_bit_planes=1,multi_samples=0)
        # scene.renderer.set(use_depth_peeling=True,maximum_number_of_peels=4,occlusion_ratio=0.1)
        ran_colors = np.random.random_integers(255, size=(len(cent),4))
        ran_colors[:,-1] = 255
        mypts = mlab.points3d(cent_verts[:,0],cent_verts[:,1],cent_verts[:,2],labels, 
            opacity=0.1, 
            scale_mode='none',
            scale_factor=1,
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

        # plot first and last
        def plot_pos_index(p):
            pos = cent[p]
            mlab.text3d(pos[0], pos[1], pos[2], str(p), scale=0.8)

        for p in xrange(0,len(cent-1),10):
            plot_pos_index(p)
        plot_pos_index(len(cent)-1)

        scene.disable_render = False

    DATADF.to_csv('_'.join([filebase,SCALAR_NAME,'rawdata.csv']), index=False)

    mg = plt.get_current_fig_manager()
    # mg.resize(*mg.window.maxsize())
    plt.savefig('{}.pdf'.format('_'.join([filebase,SCALAR_NAME])), dpi=300)
    plt.show(block=False)
    mlab.show()

if __name__ == '__main__':
    main()

