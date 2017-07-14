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

from gts.gtsconfig import GtsConfig


def stats_per_group(x):
    sid_count = len(x.sid.unique())

    x = x.groupby(['sid'])
    x = x.value.mean()

    return pd.DataFrame({'values': [val for val in x], 'subj_count': [sid_count for i in x]})


def qtile_delta(x):
    lower_quartile, upper_quartile = np.percentile(x, [25, 75])
    res = abs(upper_quartile - lower_quartile)
    return res


def plot(df, options):

    UNIQ_GROUPS = df.group.unique()
    UNIQ_GROUPS.sort()

    sns.set_style("white")
    grppal = sns.color_palette("Set2", len(UNIQ_GROUPS))

    print '# UNIQ GROUPS', UNIQ_GROUPS

    cent_stats = df.groupby(['position', 'group']).apply(stats_per_group)
    cent_stats.reset_index(inplace=True)
    print cent_stats

    import time
    from sklearn import preprocessing
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ExpSineSquared, ConstantKernel, RBF

    mean = cent_stats['values'].mean()

    # kernel = ConstantKernel() + Matern(length_scale=mean, nu=3 / 2) + \
        # WhiteKernel(noise_level=1e-10)

    kernel = 1**2 * Matern(length_scale=1, nu=1.5) + \
        WhiteKernel(noise_level=0.1)

    figure = plt.figure(figsize=(10, 6))

    palette = sns.color_palette('muted')
    for i,GRP in enumerate(UNIQ_GROUPS):
        groupDf = cent_stats[cent_stats['group'] == GRP]
        X = groupDf['position'].values.reshape((-1, 1))

        y = groupDf['values'].values.reshape((-1, 1))
        y = preprocessing.scale(y)
        
        N = groupDf['subj_count'].values.max()

        # sns.lmplot(x="position", y="values", row="group",
        #            fit_reg=False, data=groupDf)

        stime = time.time()
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, y)
        print gp.kernel_
        print gp.log_marginal_likelihood()

        print("Time for GPR fitting: %.3f" % (time.time() - stime))

        stime = time.time()

        pred_x = np.linspace(0, 30, 100)
        y_mean, y_std = gp.predict(pred_x.reshape((-1, 1)), return_std=True)
        y_mean = y_mean[:, 0]
        print("Time for GPR prediction: %.3f" % (time.time() - stime))

        group_color = palette[i]

        ci = y_std / math.sqrt(N) * 1.96
        plt.scatter(X, y, color=group_color, alpha=0.1)
        plt.plot(pred_x, y_mean, color=group_color)
        plt.fill_between(pred_x, y_mean - ci, y_mean +
                         ci, color=group_color, alpha=0.3)

    if options.title:
        plt.suptitle(options.title)

    if options.output:
        plt.savefig(options.output, dpi=150)

    if options.is_show:
        plt.show()


def main():
    parser = OptionParser(usage="Usage: %prog [options] statsfile")
    parser.add_option('-t', '--title', dest='title')
    parser.add_option('-o', '--output', dest='output')
    parser.add_option('-c', '--config', dest='config')
    parser.add_option('--yrange', dest='yrange')
    parser.add_option('--xrange', dest='xrange')
    parser.add_option('--reverse', dest='is_reverse', action='store_true',
                      default=False, help='Reverse the centroid measure stepping order')
    parser.add_option('--noshade', dest='is_shade',
                      action='store_false', default=True)
    parser.add_option('--no-show', action='store_false',
                      dest='is_show', default=True)
    parser.add_option('--annot', dest='annot')

    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(2)

    DF = pd.read_csv(args[0])

    plot(DF, options)


if __name__ == '__main__':
    main()
