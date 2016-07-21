#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
import math
import sys
import matplotlib.colors as colors
import matplotlib.cm as mcmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from optparse import OptionParser
import os.path as path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import stats

from fiber_stats import stats_per_group, position_stats

def heat_stats_per_group(x):
    res = {}
    res['mean'] = np.average(x)
    res['stdev'] = np.std(x)
    return res

def complex_array_to_rgb(X, theme='dark', rmax=None):
    '''Takes an array of complex number and converts it to an array of [r, g, b],
    where phase gives hue and saturaton/value are given by the absolute value.
    Especially for use with imshow for complex plots.'''
    absmax = rmax or np.imag(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.real(X) / np.real(X).max()
    if theme == 'light':
        Y[..., 1] = np.clip(np.imag(X) / absmax, 0, 1)
        Y[..., 2] = 1
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.imag(X) / absmax, 0, 1)
    Y = colors.hsv_to_rgb(Y)
    return Y

def plot(df, options):
    UNIQ_GROUPS = df.group.unique()
    UNIQ_GROUPS.sort()

    # prepare the plt plot
    # len_cent = len(UNIQ_CENTROIDS)

    # pal = sns.color_palette("bright", len_cent)
    sns.set_style("white")
    grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

    print '# UNIQ GROUPS',UNIQ_GROUPS


    """ 
        plot each group by their position 
    """

    fig = plt.figure(figsize=(14,7))
    cur_axe = plt.gca()

    plt.xlabel('Position Index')     
    # cent_patch = mpatches.Patch(color=cent_color, label='Centroid {}'.format(ci+1))
    # cent_legend = axes[0].legend(handles=[cent_patch], loc=9)
    # axes[0].add_artist(cent_legend)

    """
        Perform stats
    """   

    # if len(UNIQ_GROUPS) > 1:
    #     pvalsDf = position_stats(df)
    #     logpvals = np.log(pvalsDf)*-1
    #     #print logpvals


    #     pvals = logpvals.mask(pvalsDf >= 0.05 ) 
    # print df
    meanDF = df.groupby(['position','group']).value.apply(lambda x: np.median(x))
    meanDF = meanDF.unstack()

    stdDF = df.groupby(['position','group']).value.apply(lambda x: np.std(x))
    stdDF = stdDF.unstack()

    # construct hsv table
    HSV = np.zeros(meanDF.shape + (3,), dtype='float')
    normMean = meanDF / meanDF.max()
    normStd = 1-(stdDF / stdDF.max())

    # print normMean
    import matplotlib.cm as cm
    # colorfunc = np.vectorize(lambda x: cm.viridis(x)[:3])
    RGB = np.array([ cm.viridis(i)[:3] for i in normMean.values.flatten()])
    RGB = RGB.reshape(meanDF.values.shape+(3,))
    HSV = colors.rgb_to_hsv(RGB)
    HSV[...,2] *= np.clip(normStd, 0.2, 1)

    print HSV
    RGB = colors.hsv_to_rgb(HSV)
    # complexDF = meanDF + 1j*stdDF
    # print complexDF
    # legend_handles = []
    # plotDF = None
    # for gi, GRP in enumerate(UNIQ_GROUPS):
    #     subgrp = df[df['group']==GRP]

    #     posGrp = subgrp.groupby('position', sort=True)
    #     cent_stats = posGrp.value.apply(lambda x: heat_stats_per_group(x))
    #     # print cent_stats
    #     cent_stats = cent_stats.unstack()
    #     # cent_median_scalar = cent_stats['median'].tolist()
    #     # # x = [i for i in posGrp.groups]
    #     # # print x
    #     print cent_stats


    # import matplotlib.ticker as mticker
    # # print pvals
    # # cmap = mcmap.Reds
    # # cmap.set_bad('w',1.)
    # cur_axe.pcolormesh(plotDF.values.T,cmap='viridis', edgecolors='none')
    cur_axe.imshow(np.transpose(RGB,(1,0,2)),interpolation='nearest',aspect='auto')
    #cur_axe.yaxis.set_major_locator(mticker.MultipleLocator(base=1.0))
    # cur_axe.set_yticks(np.arange(pvals.values.shape[1])+0.5, minor=False)
    # cur_axe.set_yticklabels(pvalsDf.columns.values.tolist(), minor=False)
        # print cent_stats['median'].tolist()
    #     mcolor = np.array(grppal[gi])
    #     # if gi>0:
    #     #     mcolor*= 1./(1+gi)

    #     # cent_color = tuple(cent_color)
    #     mcolor = tuple(mcolor)


    #     cur_axe.set_ylabel('Scalar')
    #     # cur_axe.yaxis.label.set_color(cent_color)
    #     # cur_axe.tick_params(axis='y', colors=cent_color)

    #     #cur_axe.fill_between(x, [s[0] for s in cent_ci], [t[1] for t in cent_ci], alpha=0.3, color=mcolor)

    #     cur_axe.fill_between(x, [s[0] for s in cent_stats['whisk'].tolist()], 
    #         [t[1] for t in cent_stats['whisk'].tolist()], alpha=0.1, color=mcolor)

    #     cur_axe.fill_between(x, [s[0] for s in cent_stats['qtile'].tolist()], 
    #         [t[1] for t in cent_stats['qtile'].tolist()], alpha=0.4, color=mcolor)

    #     cur_axe.errorbar(x, cent_stats['median'].tolist(), yerr=[[s[0] for s in cent_stats['err'].tolist()], 
    #         [t[1] for t in cent_stats['err'].tolist()]], color=mcolor, alpha=0.1)    

    #     hnd, = cur_axe.plot(x,cent_stats['median'].tolist(), c=mcolor)   
    #     legend_handles.append(hnd) 

    #     # cur_axe.scatter(x,cent_stats['median'].tolist(), c=mcolor)   

    #     if options.xrange:
    #         plotrange = options.xrange.split(',')
    #         cur_axe.set_xlim([int(plotrange[0]), int(plotrange[1])])

    #     if options.yrange:
    #         plotrange = options.yrange.split(',')
    #         cur_axe.set_ylim([float(plotrange[0]), float(plotrange[1])])

    # cur_axe.legend(legend_handles, UNIQ_GROUPS)
    plt.show()

def main():
    parser = OptionParser(usage="Usage: %prog [options] statsfile")
    parser.add_option('--reverse', dest='is_reverse', action='store_true', default=False, help='Reverse the centroid measure stepping order')
    parser.add_option('--xrange', dest='xrange')
    parser.add_option('--yrange', dest='yrange')
    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(2)

    DF = pd.read_csv(args[0])

    plot(DF, options)


if __name__ == '__main__':
    main()
