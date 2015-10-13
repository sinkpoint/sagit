#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import gts.maps.tract_density as tdm
import sys

from optparse import OptionParser

_DEBUG = 0  


def run(args):
    outbasename = args.out.split('.')[0]
    tdm.tracts_to_density(args.ref, args.in_fiber, outbasename)

def main(args=None):
    parser = OptionParser(usage="Converts vtk tracts to nifti density image")
    parser.add_option("-f", "--in_fiber", dest="in_fiber",help="Input fiber bundle in vtk")
    parser.add_option("-m", "--mask", dest="mask",help="Integer binary mask")
    parser.add_option("-r", "--reference", dest="ref",help="Reference image to use as image space")
    parser.add_option("-s", "--resolution", dest="res",help="If no reference image, then the resolution of voxel, default=1")
    parser.add_option("-o", "--output", dest="out",help="Output nifti filename")

    (options, args) =  parser.parse_args()

    if not options.in_fiber or not options.reference or options.output:
        parser.print_help()
        sys.exit(2)

    run(options)


if __name__ == '__main__':
    main()