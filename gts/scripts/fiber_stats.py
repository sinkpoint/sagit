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
import scikits.bootstrap as bootstrap
import scipy.stats as stats
import yaml

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
from gts.gtsconfig import GtsConfig

BOOTSTRAP_NUM = 200

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

def _ttest_finish(df,t):
    from scipy.stats import distributions
    """Common code between all 3 t-test functions."""
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail
    if t.ndim == 0:
        t = t[()]

    return t, prob

def _chk2_asarray(a, b, axis):
    if axis is None:
        a = np.ravel(a)
        b = np.ravel(b)
        outaxis = 0
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        outaxis = axis
    return a, b, outaxis
    
def ttest_ind_log(a, b, axis=0, equal_var=True):

    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.
    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances.
    CHANGE: The number of samples N is calculated as log(N), to be used for very large number of samples

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        Axis can equal None (ravel array first), or an integer (the axis
        over which to operate on a and b).
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test
        that assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.
        .. versionadded:: 0.11.0
    Returns
    -------
    t : float or array
        The calculated t-statistic.
    prob : float or array
        The two-tailed p-value.
    
    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)

    v1 = np.var(a, axis, ddof=1)
    v2 = np.var(b, axis, ddof=1)
    n1 = math.log(a.shape[axis])
    n2 = math.log(b.shape[axis])

    if (equal_var):
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, axis) - np.mean(b, axis)
    t = np.divide(d, denom)
    t, prob = _ttest_finish(df, t)

    return t, prob

def resample_data(df, num_sample_per_pos = 100):
    POS = df.position.unique()
    POS.sort()

    GROUPS = df.group.unique()
    GROUPS.sort()

    DF = None
    for pos in POS:
        pos_data = df[df.position==pos]

        for group in GROUPS:
            group_data = pos_data[pos_data.group==group]

            samples = np.random.choice(group_data.value, replace=True, size=num_sample_per_pos)

            table = {"group":[group]*num_sample_per_pos, "position":[pos]*num_sample_per_pos, "value":samples}

            table = pd.DataFrame(data=table)
            if DF is None:
                DF = table
            else:
                DF = pd.concat([DF, table])

    return DF

def position_stats(df, name_mapping=None):

    # print '### position stats'
    from statsmodels.stats.weightstats import ztest
    from functools32 import partial, wraps
    POS = df.position.unique()
    POS.sort()
    model = 'value ~ group'
    allpvals = None
    header = None
    DF = None

    ttest_log_wrap = wraps(partial(ttest_ind_log, equal_var=False))(ttest_ind_log)
    ttest_ind_nev = wraps(partial(stats.ttest_ind, equal_var=False))(stats.ttest_ind)
    mwu_test = wraps(partial(stats.mannwhitneyu, use_continuity=False))(stats.mannwhitneyu)

    bootstrap_sample_num = 1000
    # print df

    stats_test = ttest_ind_nev
    GROUPS = df.group.unique()
    # GROUPS = [0,3]

    for pos in POS:
        # print pos
        data = df[df.position==pos]
        data = data.groupby(['sid']).mean()
        data = resample_data(data, num_sample_per_pos=BOOTSTRAP_NUM)
        # print data
        # print data.group.unique()
        # data = df[(df.group == 0) | (df.group == 3)]
        # print data
        # sys.exit()

        #cross = smf.ols(model, data=data).fit()
        #anova = sm.stats.anova_lm(cross, type=1)
        # print data.group

        mcp = MultiComparison(data.value, data.group.astype(int))

        rtp = mcp.allpairtest(stats_test, method='bonf')
        mheader = []
        for itest in rtp[2]:
            name1 = itest[0]
            name2 = itest[1]
            if name_mapping is not None:
                name1 = name_mapping[str(name1)]
                name2 = name_mapping[str(name2)]

            mheader.append("{} - {}".format(name1, name2))

        if not header or len(mheader) > len(header):
            header = mheader
        
        # get the uncorrecte pvals
        pvals = rtp[1][0][:,1]

        ndf = pd.DataFrame(data=[pvals], columns=mheader)
        if allpvals is None:
            allpvals = ndf
        else:
            allpvals = pd.concat([allpvals,ndf])
    
    # return allpvals
    # corr_pvals = allpvals    
    # print allpvals
    # return allpvals

    flatten = allpvals.values.ravel()
    flatten = flatten * 2
    mcpres = multipletests(flatten, alpha=0.05, method='bonf')
    # print mcpres
    corr_pvals = np.array(mcpres[1])
    # print corr_pvals
    corr_pvals = np.reshape(corr_pvals, (len(POS),-1))

    # print corr_pvals,corr_pvals.shape,header
    data = pd.DataFrame(data=corr_pvals, columns=header)
    data = data[data.columns[:3]]
    return data

def stats_per_group(x):
    print 'stats-per-group'

    x = x.groupby(['sid']).mean()
    x = x.value

    print len(x)

    res = {'median':[],'qtile':[]}
    medians = np.median(x)
    res['mean'] = np.average(x)
    res['median'] = medians
    lower_quartile, upper_quartile = np.percentile(x, [25,75])
    res['qtile'] = (upper_quartile, lower_quartile)
    # res['ci'] = np.percentile(x, [2.5,97.5])
    iqr = upper_quartile - lower_quartile
    upper_whisker = x[x<=upper_quartile+1.5*iqr].max()
    lower_whisker = x[x>=lower_quartile-1.5*iqr].min()
    res['whisk'] = (lower_whisker, upper_whisker)
    res['err'] = (np.abs(lower_whisker-medians), np.abs(upper_whisker-medians))

    res['ci'] = bootstrap.ci(x, n_samples=BOOTSTRAP_NUM)

    return pd.Series(res)

def smooth(X, Y):
    # ####### don't smooth    
    # return (X, Y.tolist())
    """
        Smoothing function 
    """
    from scipy.interpolate import spline
    X_new = np.linspace(X.min(),X.max(),300)
    smoothed = spline(X,Y.tolist(),X_new)
    return (X_new, smoothed)

def main():
    parser = OptionParser(usage="Usage: %prog [options] <tract.vtp>")
    parser.add_option("-s", "--scalar", dest="scalar", default="FA", help="Scalar to measure")
    parser.add_option("-n", "--num", dest="num", default=50, type='int', help="Number of subdivisions along centroids")
    parser.add_option("-l", "--local", dest="is_local", action="store_true", default=False, help="Measure from Quickbundle assigned streamlines. Default is to measure from all streamlines")
    parser.add_option("-d", "--dist", dest="dist", default=20, type='float', help="Quickbundle distance threshold")
    parser.add_option("--curvepoints", dest="curvepoints_file", help="Define a curve to use as centroid. Control points are defined in a csv file in the same space as the tract points. The curve is the vtk cardinal spline implementation, which is a catmull-rom spline.")
    parser.add_option('--yrange', dest='yrange')
    parser.add_option('--xrange', dest='xrange')
    parser.add_option('--reverse', dest='is_reverse', action='store_true', default=False, help='Reverse the centroid measure stepping order')
    parser.add_option('--pairplot', dest='pairplot',)
    parser.add_option('--noviz',dest='is_viz', action='store_false', default=True)
    parser.add_option('--config', dest='config')
    parser.add_option('--background', dest='bg_file', help='Background NIFTI image')
    parser.add_option('--annot', dest='annot')
    
    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(2)

    name_mapping = None
    if options.config:
        config = GtsConfig(options.config, configure=False)
        name_mapping = config.group_labels

    annotations = None
    if options.annot:
        with open(options.annot,'r') as fp:
            annotations = yaml.load(fp)
            

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
        stream_pids.append(i)
        stream_sids.append(subjects[i])
        try:
            stream_groups.append(groups[i])
        except Exception:
            # group might not exist
            pass

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


    if options.is_viz:
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

        # UNIQ_GROUPS = [0,1]

        grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

        print '# UNIQ GROUPS',UNIQ_GROUPS


        # print df
        # df = df[df['sid'] != 15]
        # df = df[df['sid'] != 16]
        # df = df[df['sid'] != 17]
        # df = df[df['sid'] != 18]
        """ 
            plot each group by their position 
        """

        fig = plt.figure(figsize=(14,7))
        ax1 = plt.subplot2grid((4,3),(0,0),colspan=3,rowspan=3)
        ax2 = plt.subplot2grid((4,3),(3,0),colspan=3,sharex=ax1)
        axes = [ax1,ax2]

        plt.xlabel('Position Index')   
        
        if len(centroids) > 1:  
            cent_patch = mpatches.Patch(color=cent_color, label='Centroid {}'.format(ci+1))
            cent_legend = axes[0].legend(handles=[cent_patch], loc=9)
            axes[0].add_artist(cent_legend)

        """
            Perform stats
        """   

        if len(UNIQ_GROUPS) > 1:
            # df = resample_data(df, num_sample_per_pos=120)
            # print df
            pvalsDf = position_stats(df, name_mapping=name_mapping)
            logpvals = np.log(pvalsDf)*-1
            # print logpvals


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
            print '-------------------- GROUP ',gi,'----------------------'
            subgrp = df[df['group']==GRP]
            print len(subgrp)
            
            if options.xrange:
                x0, x1 = options.xrange.split(',')
                x0 = int(x0)
                x1 = int(x1)
                subgrp = subgrp[(subgrp['position'] >= x0) & (subgrp['position'] < x1)]

            posGrp = subgrp.groupby('position', sort=True)
            

            cent_stats = posGrp.apply(lambda x:stats_per_group(x))

            if len(cent_stats) == 0:
                continue
            
            cent_stats = cent_stats.unstack()
            cent_median_scalar = cent_stats['median'].tolist()

            x = np.array([i for i in posGrp.groups])
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

            # cur_axe.fill_between(x, [s[0] for s in cent_stats['whisk'].tolist()], 
            #     [t[1] for t in cent_stats['whisk'].tolist()], alpha=0.1, color=mcolor)

            qtile_top = np.array([s[0] for s in cent_stats['ci'].tolist()])
            qtile_bottom = np.array([t[1] for t in cent_stats['ci'].tolist()])

            x_new, qtop_sm = smooth(x, qtile_top)
            x_new, qbottom_sm = smooth(x, qtile_bottom)
            cur_axe.fill_between(x_new, qtop_sm, qbottom_sm 
                , alpha=0.25, color=mcolor)

            # cur_axe.errorbar(x, cent_stats['median'].tolist(), yerr=[[s[0] for s in cent_stats['err'].tolist()], 
            #     [t[1] for t in cent_stats['err'].tolist()]], color=mcolor, alpha=0.1)    

            x_new, median_sm = smooth(x, cent_stats['median'])
            hnd, = cur_axe.plot(x_new, median_sm, c=mcolor)   
            legend_handles.append(hnd) 

            # cur_axe.scatter(x,cent_stats['median'].tolist(), c=mcolor)

            if options.yrange:
                plotrange = options.yrange.split(',')
                cur_axe.set_ylim([float(plotrange[0]), float(plotrange[1])])

        
        legend_labels = UNIQ_GROUPS
        if name_mapping is not None:
            legend_labels = [ name_mapping[str(i)] for i in UNIQ_GROUPS]
        cur_axe.legend(legend_handles, legend_labels)

        if annotations:
            for key,val in annotations.iteritems():
                # print key
                cur_axe.axvspan(val[0],val[1],fill=False, linestyle='dashed')
                axis_to_data = cur_axe.transAxes + cur_axe.transData.inverted()
                data_to_axis = axis_to_data.inverted()
                axpoint = data_to_axis.transform((val[0],0))
                # print axpoint
                cur_axe.text(axpoint[0], 1.02, key, transform=cur_axe.transAxes)

        """
            Plot 3D Viz 
        """

        if options.is_viz:
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
    outfile = '_'.join([filebase, SCALAR_NAME])
    print 'save to {}'.format(outfile)
    plt.savefig('{}.pdf'.format(outfile), dpi=300)

    if options.is_viz:
        plt.show(block=False)
        mlab.show()

if __name__ == '__main__':
    main()

