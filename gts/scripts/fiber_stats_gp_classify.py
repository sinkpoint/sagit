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

    cent_stats = df.groupby(
        ['position', 'group', 'side']).apply(stats_per_group)
    cent_stats.reset_index(inplace=True)

    import time
    from sklearn import preprocessing
    from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ExpSineSquared, ConstantKernel, RBF


    ctlDF = cent_stats[ cent_stats['group'] == 0 ]

    TNRightDF = cent_stats[ cent_stats['group'] != 0]
    TNRightDF = TNRightDF[TNRightDF['side'] == 'right']

    dataDf = pd.concat([ctlDF, TNRightDF], ignore_index=True)
    print dataDf

    yDf = dataDf['group'] == 0
    yDf = yDf.astype(int)
    y = yDf.values
    print y
    print y.shape

    XDf = dataDf[['position', 'values']]
    X = XDf.values
    X = preprocessing.scale(X)
    print X
    print X.shape
    

    # kernel = ConstantKernel() + Matern(length_scale=mean, nu=3 / 2) + \
    # WhiteKernel(noise_level=1e-10)
    
    kernel = 1**2 * Matern(length_scale=1, nu=1.5) + \
        WhiteKernel(noise_level=0.1)

    figure = plt.figure(figsize=(10, 6))


    stime = time.time()
    gp = GaussianProcessClassifier(kernel)
    gp.fit(X, y)

    print gp.kernel_
    print gp.log_marginal_likelihood()

    print("Time for GPR fitting: %.3f" % (time.time() - stime))


    # create a mesh to plot in
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    plt.figure(figsize=(10, 5))
    
    # Plot the predicted probabilities. For that, we will assign a color to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].

    Z = gp.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:,1]
    print Z
    print Z.shape
    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1]))
    print Z.shape
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g"])[y])
    plt.xlabel('position')
    plt.ylabel('normalized val')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("%s, LML: %.3f" %
            ("TN vs. Control", gp.log_marginal_likelihood(gp.kernel_.theta)))

    plt.tight_layout()


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
