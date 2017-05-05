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
import yaml

from fiber_stats import stats_per_group, position_stats
from gts.gtsconfig import GtsConfig


def heat_stats_per_group(x):
    res = {}
    res['mean'] = np.average(x)
    res['stdev'] = np.std(x)
    lower_quartile, upper_quartile = np.percentile(x, [25,75])
    res['qtile'] = abs(upper_quartile-lower_quartile)
    return res


def qtile_delta(x):
    lower_quartile, upper_quartile = np.percentile(x, [25,75])
    res = abs(upper_quartile-lower_quartile)
    return res


def plot(df, options):

    UNIQ_GROUPS = df.group.unique()
    UNIQ_GROUPS.sort()

    sns.set_style("white")
    grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

    print '# UNIQ GROUPS',UNIQ_GROUPS


    """ 
        plot each group by their position 
    """

    fig = plt.figure(figsize=(14,7))
    cur_axe = plt.gca()

    plt.xlabel('Position')

    """
        Perform stats
    """   

    meanDF = df.groupby(['position','group']).value.apply(lambda x: np.median(x))
    meanDF = meanDF.unstack()

    print meanDF
    stdDF = df.groupby(['position','group']).value.apply(lambda x: np.std(x))
    stdDF = stdDF.unstack()

    qtileDF = df.groupby(['position','group']).value.apply(lambda x: qtile_delta(x))
    qtileDF = qtileDF.unstack()

    # construct hsv table
    HSV = np.zeros(meanDF.shape + (3,), dtype='float')
    normMean = meanDF / meanDF.max()
    normStd = 1-(stdDF / stdDF.max())
    normQtile = 1-(qtileDF / qtileDF.max())

    # print normMean
    import matplotlib.cm as cm
    RGB = np.array([ cm.viridis(i)[:3] for i in normMean.values.flatten()])
    RGB = RGB.reshape(meanDF.values.shape+(3,))
    HSV = colors.rgb_to_hsv(RGB)
    if options.is_shade:
        HSV[...,2] *= np.clip(normQtile, 0.3, 1) # brightness depends on qtile delta

    RGB = colors.hsv_to_rgb(HSV)
    # cur_axe.imshow(np.transpose(RGB,(1,0,2)),interpolation='nearest',aspect='auto')

    group_labels = UNIQ_GROUPS
    if options.config:
        config = GtsConfig(options.config, configure=False)   
        print config.group_labels
        group_labels = map(lambda x: config.group_labels[str(x)], UNIQ_GROUPS)
        # cur_axe.set_yticklabels(group_labels)

    sns.heatmap(meanDF.transpose(), cmap='viridis', ax=cur_axe, yticklabels=group_labels)
    # cur_axe.set_yticks(UNIQ_GROUPS)

    if options.annot:
        with open(options.annot,'r') as fp:
            annotations = yaml.load(fp)

        for key,val in annotations.iteritems():
            print key
            cur_axe.axvspan(val[0],val[1],fill=False, linestyle='dashed')
            axis_to_data = cur_axe.transAxes + cur_axe.transData.inverted()
            data_to_axis = axis_to_data.inverted()
            axpoint = data_to_axis.transform((val[0],0))
            print axpoint
            cur_axe.text(axpoint[0], 1.02, key, transform=cur_axe.transAxes)

    if options.title:
        plt.suptitle(options.title)


    if options.output:
        plt.savefig(options.output)

    if options.is_show:
        plt.ion()
        plt.show()

def main():
    parser = OptionParser(usage="Usage: %prog [options] statsfile")
    parser.add_option('-t','--title', dest='title')
    parser.add_option('-o', '--output', dest='output')
    parser.add_option('-c','--config', dest='config')
    parser.add_option('--yrange', dest='yrange')
    parser.add_option('--xrange', dest='xrange')
    parser.add_option('--reverse', dest='is_reverse', action='store_true', default=False, help='Reverse the centroid measure stepping order')
    parser.add_option('--noshade', dest='is_shade', action='store_false', default=True)
    parser.add_option('--no-show', action='store_false', dest='is_show', default=True)
    parser.add_option('--annot', dest='annot')

    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(2)

    DF = pd.read_csv(args[0])

    plot(DF, options)


if __name__ == '__main__':
    main()
