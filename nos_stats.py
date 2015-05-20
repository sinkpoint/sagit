#!/usr/bin/python

import numpy as np
import pandas as pd
from glob import glob

files = glob('stats/conf*conj*.csv')
data = None
for f in files:
    print f
    fdata = pd.io.parsers.read_csv(f)
    fdata.drop('id', 1,inplace=True)
    f_len = len(fdata)
    if 'unfiltered' in f:
        fil = np.repeat(False, f_len)
    else:
        fil = np.repeat(True, f_len)
    fdata['filtered'] = fil


    if data is None:
        data = fdata
    else:
        data = pd.concat([data, fdata])


import seaborn as sn
dpi=300
fil_data = data[data.filtered == True]
reg_order = ['vestibL','vestibR','vagusL','vagusR','redL','redR','lgnL','lgnR','mgnL','mgnR','fornix']
sn.factorplot('method', 'score', hue='region',kind='bar',aspect=2,legend_out=False, data=fil_data,palette='Paired', hue_order=reg_order)
sn.plt.savefig('nos_by_region_filtered.png', dpi=dpi)
sn.plt.show()