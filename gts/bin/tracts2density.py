#!/usr/bin/python 
# -*- coding: utf-8 -*-

import argparse
import nibabel as nib
from gts.tractDensityMap import vtkToStreamlines
from gts.tractDensityMap import getTractDensityMap
from multiprocessing import Pool

_DEBUG = 0

def run(args):
    outbasename = args.out.split('.')[0]

    ref_image = nib.load(args.ref)
    streamlines = vtkToStreamlines(args.in_fiber)

    outImage, outFibImage, outBinImage = getTractDensityMap(ref_image, streamlines)

    nib.save(outImage, args.out)
    nib.save(outFibImage, outbasename+'_fib.nii.gz')
    nib.save(outBinImage, outbasename+'_bin.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts vtk tracts to nifti density image")
    parser.add_argument("-f", "--in_fiber", required=True, dest="in_fiber",help="Input fiber bundle in vtk")
    parser.add_argument("-m", "--mask", dest="mask",help="Integer binary mask")
    parser.add_argument("-r", "--reference", dest="ref",help="Reference image to use as image space")
    parser.add_argument("-s", "--resolution", dest="res",help="If no reference image, then the resolution of voxel, default=1")
    parser.add_argument("-o", "--output", dest="out",help="Output nifti filename")

    args =  parser.parse_args()

    run(args)