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
from gts.gtsconfig import GtsConfig
import best
import best.plot

from pymc import MCMC


def heat_stats_per_group(x):
    res = {}
    res['mean'] = np.average(x)
    res['stdev'] = np.std(x)
    lower_quartile, upper_quartile = np.percentile(x, [25, 75])
    res['qtile'] = abs(upper_quartile - lower_quartile)
    return res


def qtile_delta(x):
    lower_quartile, upper_quartile = np.percentile(x, [25, 75])
    res = abs(upper_quartile - lower_quartile)
    return res


def BEST_pymc3(y1, y2):
    import pymc3 as pm

    y = pd.DataFrame(dict(value=np.r_[y1, y2], group=np.r_[
                     ['group1'] * len(y1), ['group2'] * len(y2)]))

    mu_m = y.value.mean()
    mu_s = y.value.std() * 2

    with pm.Model() as model:

        group1_mean = pm.Normal('group1_mean', mu_m, sd=mu_s)
        group2_mean = pm.Normal('group2_mean', mu_m, sd=mu_s)

    sigma_low = 1
    sigma_high = 10

    with model:

        group1_std = pm.Uniform(
            'group1_std', lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform(
            'group2_std', lower=sigma_low, upper=sigma_high)

    with model:

        nu = pm.Exponential('nu_minus_one', 1 / 29.) + 1

    with model:

        lambda1 = group1_std**-2
        lambda2 = group2_std**-2

        group1 = pm.StudentT('drug', nu=nu, mu=group1_mean,
                             lam=lambda1, observed=y1)
        group2 = pm.StudentT(
            'placebo', nu=nu, mu=group2_mean, lam=lambda2, observed=y2)

    with model:

        diff_of_means = pm.Deterministic(
            'difference of means', group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic(
            'difference of stds', group1_std - group2_std)
        effect_size = pm.Deterministic('effect size',
                                       diff_of_means / pm.sqrt((group1_std**2 + group2_std**2) / 2))

    with model:
        trace = pm.sample(500, njobs=3)

    pm.plot_posterior(trace[1000:],
                      varnames=['group1_mean', 'group2_mean',
                                'group1_std', 'group2_std', 'nu_minus_one'],
                      color='#87ceeb')

    pm.plot_posterior(trace[1000:],
                      varnames=['difference of means',
                                'difference of stds', 'effect size'],
                      ref_val=0,
                      color='#87ceeb')

    return model


def BEST(compdata, pos, iter=10000, burn=1000):
    model = best.make_model(compdata)
    M = MCMC(model)
    M.sample(iter=iter + burn, burn=burn)

    posterior_mean1 = M.trace('group1_mean')[:]
    posterior_mean2 = M.trace('group2_mean')[:]
    diff_means = posterior_mean1 - posterior_mean2
    posterior_std1 = M.trace('group1_std')[:]
    posterior_std2 = M.trace('group2_std')[:]
    diff_stds = posterior_std1 - posterior_std2
    effect_size = diff_means / \
        np.sqrt((posterior_std1**2 + posterior_std2**2) / 2)
    post_nu_minus_one = M.trace('nu_minus_one')[:]
    lognup = np.log10(post_nu_minus_one + 1)

    res = {
        'position': np.ones(posterior_mean1.shape, dtype=int) * pos,
        'diff_means': diff_means,
        'diff_stds': diff_stds,
        'effect_size': effect_size,
        'normality': lognup,
    }

    return res


def one_position_stats(compdata=None, position=0, pair=(), iter=10000, burn=1000):
    pairname = '-'.join(pair)
    print 'Position = ', position,
    print '    Pair = ', pairname

    modelstats = BEST(compdata, position, iter=iter, burn=burn)
    modelstats['compare'] = [pairname] * len(modelstats['diff_means'])

    return modelstats


def position_stats(df, iter=10000, burn=1000):
    POS = df.position.unique()
    POS.sort()

    statsDF = None

    queue = []

    for pos in POS:
        data = df[df.position == pos]
        groups = data.group.unique()
        groups.sort()
        rawdata = {}
        for i in groups:
            groupdata = data[data.group == i]
            # subjdata = groupdata.groupby('sid').mean()
            # print subjdata
            rawdata[str(i)] = groupdata.value.values

        compgroup = [
            ('0', '3'),
            # ('0','1'),
            # ('0','2'),
            # ('1','2'),
            # ('1','3')
        ]
        for pair in compgroup:
            compdata = {}
            for m in pair:
                compdata[m] = rawdata[m]

            item = {
                "pair": pair,
                "position": pos,
                "compdata": compdata,
                "iter": iter,
                "burn": burn
            }
            queue.append(item)

    print 'Queue items = ', len(queue)

    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool

    pool = ThreadPool()
    presults = pool.map(lambda x: one_position_stats(**x), queue)
    pool.close()
    pool.join()

    statsDF = pd.concat([pd.DataFrame(i) for i in presults])

    return statsDF


def GP(DF):
    import GPy

    GROUPS = DF.group.unique()
    GROUPS.sort()

    for grp in GROUPS:
        groupDF = DF[DF.group == grp]

        # Make a GP regression model
        x = groupDF.position.values
        newshape = (len(x), 1)
        print newshape
        x = x.reshape(newshape)
        y = np.array(groupDF.value.values).reshape(newshape)

        print x, y
        POS = groupDF.position.unique()
        POS.sort()
        print POS
        z = POS
        z = z.reshape((len(z), 1))

        m = GPy.models.SparseGPRegression(x, y, Z=z)
        m.optimize('bfgs')
        m.plot()
        plt.suptitle('Group {}'.format(grp))
        plt.savefig('GP_group_{}.png'.format(grp), dpi=150)


def smooth(X, Y):
    # ####### don't smooth
    # return (X, Y.tolist())
    """
        Smoothing function 
    """
    from scipy.interpolate import spline
    X_new = np.linspace(X.min(), X.max(), 300)
    smoothed = spline(X, Y.tolist(), X_new)
    return (X_new, smoothed)


def resample_data(df, num_sample_per_pos=100):
    """
        Bootstrap resample each positional data to a desired number
    """
    POS = df.position.unique()
    POS.sort()

    GROUPS = df.group.unique()
    GROUPS.sort()

    DF = None
    for pos in POS:
        pos_data = df[df.position == pos]

        for group in GROUPS:
            group_data = pos_data[pos_data.group == group]

            samples = np.random.choice(
                group_data.value, replace=True, size=num_sample_per_pos)

            table = {"group": [group] * num_sample_per_pos,
                     "position": [pos] * num_sample_per_pos, "value": samples}

            table = pd.DataFrame(data=table)
            if DF is None:
                DF = table
            else:
                DF = pd.concat([DF, table])

    return DF


def main():
    parser = OptionParser(usage="Usage: %prog [options] statsfile")
    parser.add_option('--reverse', dest='is_reverse', action='store_true',
                      default=False, help='Reverse the centroid measure stepping order')
    parser.add_option('--xrange', dest='xrange')
    parser.add_option('--yrange', dest='yrange')
    parser.add_option('--config', dest='config')
    parser.add_option('-f', '--fromfile', dest='statsfile')
    parser.add_option('-o', '--output', dest='output')
    parser.add_option('--no-show', action='store_false',
                      dest='is_show', default=True)
    parser.add_option('--iter', dest='iter', default=10000, type='int')
    parser.add_option('--burn', dest='burn', default=1000, type='int')

    (options, args) = parser.parse_args()

    if not options.statsfile:
        if len(args) == 0:
            parser.print_help()
            sys.exit(2)

        basename = path.splitext(args[0])[0]
        DF = pd.read_csv(args[0])

        # GP(DF)
        # plt.show()

        DF = resample_data(DF, num_sample_per_pos=100)
        statsDF = position_stats(DF, iter=options.iter, burn=options.burn)
        print statsDF

        from best import calculate_sample_statistics

        plotDF = None

        COMPGROUP = statsDF.compare.unique()

        for ci, comparison in enumerate(COMPGROUP):
            compdata = statsDF[statsDF.compare == comparison]
            print compdata
            POS = compdata.position.unique()
            POS.sort()

            compStatsDF = None
            for pos in POS:
                data = compdata[compdata.position == pos]

                stats = calculate_sample_statistics(data['diff_means'])
                stats['position'] = pos
                stats['compare'] = comparison

                # convert all values to list for pandas
                for k, v in stats.iteritems():
                    stats[k] = [v]

                # print stats
                posDF = pd.DataFrame(stats)
                if compStatsDF is None:
                    compStatsDF = posDF
                else:
                    compStatsDF = pd.concat([compStatsDF, posDF])

            if plotDF is None:
                plotDF = compStatsDF
            else:
                plotDF = pd.concat([plotDF, compStatsDF])

        plotDF.to_csv('{}_stats.csv'.format(basename), index_label='index')
        # statsDF.to_csv('{}_stats.csv'.format(basename), index_label='index')
    else:
        basename = path.splitext(options.statsfile)[0]
        plotDF = pd.read_csv(options.statsfile, index_col=0)

    print plotDF

    COMPGROUP = plotDF.compare.unique()
    pal = sns.color_palette()

    # ax = sns.violinplot(x='position', y='diff_means', hue='compare', inner='box', data=statsDF)
    num_axes = len(COMPGROUP)
    axheight = 5 * num_axes
    fig, axes = sns.plt.subplots(
        num_axes, sharex=True, figsize=(10, int(axheight)))
    delta = 0.5
    for ci, comparison in enumerate(COMPGROUP):
        curax = axes[ci]
        gcolor = pal[ci]
        compdata = plotDF[plotDF.compare == comparison]

        POS = compdata['position']
        POS_new, mean_sm = smooth(POS, compdata['mean'])
        POS_new, hid_min_sm = smooth(POS, compdata['hdi_min'])
        POS_new, hid_max_sm = smooth(POS, compdata['hdi_max'])

        curax.fill_between(POS_new, hid_min_sm, hid_max_sm,
                           alpha=0.1, color=gcolor)

        curax.plot(POS_new, hid_max_sm, ':', color=gcolor, alpha=0.5)
        curax.plot(POS_new, hid_min_sm, ':', color=gcolor, alpha=0.5)
        curax.plot(POS_new, mean_sm, color=gcolor)

        curax.axhline(y=0., c="red", linewidth=0.5)
        curax.set_ylim([-delta, delta])

    # sns.plt('')
    # sns.tsplot(time='position', value='diff_means', ci=95, condition='compare', data=statsDF)
    outputfile = '{}_compare_plot.png'.format(basename)
    if options.output:
        outputfile = options.output

    sns.plt.savefig(outputfile, dpi=150)
    if (options.is_show):
        sns.plt.show()
    # plot(DF, options)


if __name__ == '__main__':
    main()
