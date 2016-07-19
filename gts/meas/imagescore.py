#!/usr/bin/env python

import nibabel as nib
import numpy as np
import sys

from math import log
from scipy.optimize import curve_fit

def main(): 
    if len(sys.argv) < 2: 
        print 'Usage: {0} <input.nii.gz> <figure.(png)|(svg)|(pdf)>'.format(sys.argv[0])
        return

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        image = sys.argv[2]
    else:
        image = True
    from_file(filename, image)

def from_file(filename, figure=True, title=None):
    img = nib.load(filename)
    data = img.get_data()
    return get_score(data, figure=figure, title=title)

def get_score(data, figure=True, title=None):
    vals = data.ravel()

    min = np.min(vals)

    try:
        max_val = log(len(vals[vals>0]))
    except ValueError:
        max_val = 0
        print 'SCORE 0'
        return 0


    n_bins = 10
    step = 1.0/n_bins

    thres_bin = range(0,n_bins)
    thres_vals = []

    for i in thres_bin:	
        thres = float(i)*step
        if thres>0:
            tvals = vals[vals>=thres]
        else:
            tvals = vals[vals>thres]

        try:
            val = log(len(tvals))
        except ValueError:
            val = 0

        val = val / max_val
        thres_vals.append(val)

    print thres_bin
    thres_bin_labels = [ i*0.1 for i in thres_bin]
    print thres_vals
    score = np.sum(np.multiply(thres_vals, step))
    print 'SCORE',score

    if figure:
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.set_style('white')
        sns.set_context("notebook", font_scale=1.8)

        mcolor = sns.color_palette()[0]
        pcoef = np.polyfit(thres_bin, thres_vals, 3)
        curve = np.poly1d(pcoef)
        fig = plt.figure(figsize=(6,5))        
        ax = plt.subplot()
        if not title:
            title = 'NOS Score'
        ax.set_title(title+' (score=%1.3f)' % score )
        plt.xlabel('Overlap Threshold')
        plt.ylabel('log(count) ratio')
        plt.plot(thres_bin_labels, thres_vals, 'ks-', color=mcolor)
        plt.axhline(0, color=mcolor)
        plt.fill_between(thres_bin_labels, 0, thres_vals, facecolor=mcolor, alpha=0.5)
        fig.tight_layout()
        #curve_fit(f, xdata, ydata)
        if type(figure) is str or type(figure) is unicode:
            plt.savefig(figure)
        else:
            plt.show()
        plt.close()


    return score


if __name__ == '__main__':  
    main()