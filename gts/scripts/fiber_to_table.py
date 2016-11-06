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
from scipy import stats

from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd
from optparse import OptionParser
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking import metrics as tm
from scipy.cluster.vq import kmeans2
import os.path as path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests



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
    parser = OptionParser(usage="Usage: %prog [options] <tract.vtp> <output.csv>")
    parser.add_option("-d", "--dist", dest="dist", default=20, type='float', help="Quickbundle distance threshold")
    parser.add_option("-n", "--num", dest="num", default=50, type='int', help="Number of subdivisions along centroids")
    parser.add_option("-s", "--scalar", dest="scalar", default="FA", help="Scalar to measure")
    parser.add_option("--curvepoints", dest="curvepoints_file", help="Define a curve to use as centroid. Control points are defined in a csv file in the same space as the tract points. The curve is the vtk cardinal spline implementation, which is a catmull-rom spline.")
    parser.add_option("-l", "--local", dest="is_local", action="store_true", default=False, help="Measure from Quickbundle assigned streamlines. Default is to measure from all streamlines")
    parser.add_option('--reverse', dest='is_reverse', action='store_true', default=False, help='Reverse the centroid measure stepping order')
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
        step = (param_range[1]-param_range[0])/(QB_NPOINTS)

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
        # cent_color = np.array(pal[ci])

        c, labels = kmeans2(cent_verts, cent, iter=1)

        cid = np.ones(len(labels))
        d = {'value':cent_scalars, 'position':labels, 'group':cent_groups, 'pid':cent_pids, 'sid':cent_sids, 'centroid':ci}

        df = pd.DataFrame(data=d)
        if DATADF is None:
            DATADF = df
        else:
            pd.concat([DATADF, df])


    outfilename = '_'.join([filebase,SCALAR_NAME,'rawdata.csv'])
    if len(args) > 1:
        outfilename = args[1]

    if outfilename.endswith('.csv'):
        DATADF.to_csv(outfilename, index=False)

    if outfilename.endswith('.xls'):
        DATADF.to_excel(outfilename, index=False)

if __name__ == '__main__':
    main()

