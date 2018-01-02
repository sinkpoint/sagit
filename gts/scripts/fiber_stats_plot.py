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
import os.path as path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import stats
import yaml

from fiber_stats import stats_per_group, position_stats
from gts.gtsconfig import GtsConfig

args = None

def heat_stats_per_group(x):
    x = x.groupby(['sid']).mean()
    x = x.value
    
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


def annotate_group(name, vspan, ax=None):
    """Annotates a span of the x-axis"""
    def annotate(ax, name, left, right, y, pad):
        arrow = ax.annotate(name,
                xy=(y, left), xycoords='data',
                xytext=(y-pad, right), textcoords='data',
                annotation_clip=False, verticalalignment='top',
                horizontalalignment='center', linespacing=2.0,
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0,
                                connectionstyle='angle,angleA=90,angleB=0,rad=5')
                )
        return arrow

    if ax is None:
        ax = plt.gca()
        
    axis_limit = ax.get_xlim()[0]
    pad = 0.04 * np.ptp(ax.get_ylim())
    vcenter = np.mean(vspan)
    left_arrow = annotate(ax, name, vspan[0], vcenter, axis_limit, pad)
    right_arrow = annotate(ax, name, vspan[1], vcenter, axis_limit, pad)
    return left_arrow, right_arrow

def plot(df, options):
    
    df['group'] = df['group'].apply(str)
    df['joint'] = df.group.str.cat(df.side)

    selectCol = options.groupby

    UNIQ_GROUPS = df[selectCol].unique()
    UNIQ_GROUPS.sort()

    sns.set_style("white")
    sns.set_context("talk")
    grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

    print '# UNIQ GROUPS',UNIQ_GROUPS


    """ 
        plot each group by their position 
    """

    fig = plt.figure(figsize=(5,15))
    cur_axe = plt.gca()

    plt.xlabel('Group',size=20)
    plt.ylabel('Position',size=20)
    ylabels = None

    """
        Perform stats
    """   

    # df = df[ df.position > 0 ]
    meanDF = df.groupby(['position', selectCol]
                        ).value.apply(lambda x: np.mean(x))
    meanDF = meanDF.unstack()
    print meanDF
    # stdDF = df.groupby(['position','group']).value.apply(lambda x: np.std(x))
    # stdDF = stdDF.unstack()

    # qtileDF = df.groupby(['position','group']).value.apply(lambda x: qtile_delta(x))
    # qtileDF = qtileDF.unstack()


    group_labels = UNIQ_GROUPS

    if options.orient == 'H':
        meanDF = meanDF.transpose()
        plt.xlabel('Position')
        ylabels = group_labels
        
    # cbar_kws = {'label': args.scalar}
    # cbar_kws = {'aspect': 20, 'fraction': 0.1}
    cbar_kws = {}

    measureDf = meanDF[meanDF.index > 0]

    vmax = measureDf.max().max()
    vmin = measureDf.min().min()

    print vmin,vmax

    ax = sns.heatmap(meanDF, cmap='viridis', ax=cur_axe, vmin=vmin, vmax=vmax,
                     square=False, cbar_kws=cbar_kws)
    ax.vlines(range(30), *ax.get_ylim(), color='white')

    # cur_axe.set_yticks(UNIQ_GROUPS)

    if options.config:
        import json
        with open(options.config, 'r') as fp:
            config = json.load(fp)
        print config['group_labels']
        group_labels = map(lambda x: config['group_labels'][str(x)], UNIQ_GROUPS)
        ax.set_xticklabels(group_labels)

    if options.annot:
        with open(options.annot,'r') as fp:
            annotations = yaml.load(fp)


        for key,val in annotations.iteritems():
            print key,val
            annotate_group(key, val, ax)

        plt.subplots_adjust(left=0.4)
        ax.yaxis.labelpad = 70

        # ax.spines['left'].set_position(('outward', 50))
    
    if options.orient == 'H':
        ax.set_yticklabels(ylabels)
    else:
        ax.invert_yaxis()

    if options.title:
        ax.set_title(options.title, size=24)

    plt.tight_layout()

    if options.output:
        plt.savefig(options.output)

    if options.is_show:
        plt.show()

def main():
    global args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='Tract')
    parser.add_argument('output', metavar='Output', nargs='?')
    parser.add_argument('-t','--title', dest='title')
    parser.add_argument('-c','--config', dest='config')
    parser.add_argument('--hide', action='store_false', dest='is_show', default=True)
    parser.add_argument('--annot', dest='annot')
    parser.add_argument('--scalar', dest='scalar', default='scalar')
    parser.add_argument('--orient', dest='orient', default='V')
    parser.add_argument('--group-by', dest='groupby', default='joint')

    args = parser.parse_args()

    DF = pd.read_csv(args.input)

    plot(DF, args)


if __name__ == '__main__':
    main()
