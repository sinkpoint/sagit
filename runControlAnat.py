#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:43:13 2014

@author: dchen
"""

import sys
import os
import argparse
from gts.groupTractStats import *

conf_file = sys.argv[1]
gts =  GroupTractStats(conf_file)

#gts.preprocessDWI()
#gts.runAntsDwiToT1(bet='0.1')
gts.projectRoiT1TemplateToSingle()
gts.projectRoiT1ToDwi()
tracts = gts.seedIndividualTracts(labels=[2],recompute=False,overwrite=True)
#gts.viewTracks()
# tracts = {'mrtrix': [ u'cst_vestibR_filtered.vtk', u'cst_vestibL_filtered.vtk'],
# 			'xst' : ['vestibR_filtered.vtk','vestibL_filtered.vtk']
# }
gts.tracts_to_density(tracts)
gts.tracts_conjunction(tracts, img_type='binary')

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(prog="GTS")
#    parser.add_argument('-c','--conf', help='config file in JSON format')
#
#    antsFA2T1_sub = parser.add_subparsers(help="Runs bet steps for FA->T1 registration")
#    antsFA2T1_par = antsFA2T1_sub.add_parser('antsFa2T1')
#    antsFA2T1_par.add_argument('f', '--frac', help="bet fraction")
#    antsFA2T1_par.set_defaults(method=runAntsFaToT1)
#
#    args = parser.parse_args()
#    args.method(**vars(args))
#
#    print args





