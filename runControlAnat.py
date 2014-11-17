#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:13 2014

@author: dchen
"""

import sys
import os
import json
import argparse
from gts.groupTractStats import *

conf_file = sys.argv[1]
gts =  GroupTractStats(conf_file)

basename = os.path.basename(conf_file).split('.')[0]
tracts_file = basename+'_tracts.txt'
conj_files_list = basename+'_conj_files.txt'


# gts.preprocessDWI()
# gts.runAntsDwiToT1(bet='0.1')
# gts.projectRoiT1TemplateToSingle()
# gts.projectRoiT1ToDwi()

tracts = gts.seedIndividualTracts(labels=[2],recompute=False,overwrite=True)


with open(tracts_file, 'w') as outfile:
	json.dump(tracts, outfile, sort_keys=True, indent=4, separators=(',', ': '))

with open(tracts_file, 'r') as fp:
	tracts = json.load(fp)

print tracts
#gts.tracts_to_images(tracts)
# gts.viewTracks()
from copy import copy, deepcopy
unfiltered_tracts = deepcopy(tracts)
for k,v in unfiltered_tracts.iteritems():
    for i,j in enumerate(v):
        v[i] = j.replace('_filtered','')


def densityMapping(tracts, dry_run=False):
    # gts.tracts_to_density(tracts)

    conj_files = gts.tracts_conjunction(tracts, img_type='binary',dry_run=dry_run)
    print conj_files

    print conj_files_list
    with open(conj_files_list, 'w') as outfile:
      	json.dump(conj_files, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    with open(conj_files_list, 'r') as fp:
     	conj_files = json.load(fp)

    fig_list, basename = gts.conjunction_to_images(conj_files, name='nobg', bg_file='', dry_run=dry_run)

    gts.conjunction_images_combine(fig_list, basename=basename, group_names=['nobg'])

    fig_list, basename = gts.conjunction_to_images(conj_files, name='bg', bg_file=gts.config.orig_path+'/con_average.nii.gz', slice_indices=(128,128,80), dry_run=dry_run)
    gts.conjunction_images_combine(fig_list, basename=basename, group_names=['bg'])

print tracts
densityMapping(tracts, dry_run=False)

print unfiltered_tracts
densityMapping(unfiltered_tracts, dry_run=False)


