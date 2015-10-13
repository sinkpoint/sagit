#!/usr/bin/python

import nibabel as nib
import numpy as np
import sys

from math import log
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

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
        pcoef = np.polyfit(thres_bin, thres_vals, 3)
        curve = np.poly1d(pcoef)
        plt.figure()        
        ax = plt.subplot()
        if not title:
            title = 'normalized overlap score'
        ax.set_title(title+' (score=%1.3f)' % score )
        plt.xlabel('percent overlap threshold')
        plt.ylabel('log(count) ratio')
        plt.plot(thres_bin_labels, thres_vals, 'ks-', thres_bin_labels, curve(thres_bin), 'r--')
        plt.axhline(0, color='black')
        #curve_fit(f, xdata, ydata)
        if type(figure) is str or type(figure) is unicode:
            plt.savefig(figure)
        else:
            plt.show()
        plt.close()


    return score


if __name__ == '__main__':    
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        image = sys.argv[2]
    else:
        image = True
    from_file(filename, image)