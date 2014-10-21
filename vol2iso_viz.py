#!/usr/bin/python

from optparse import OptionParser
import numpy as np
import nibabel as nib
import sys
from mayavi import mlab
from scipy import interpolate as sinterp

def run(options, args):
	def getNiftiAsScalarField(filename):
		fimg = nib.load(filename)
		hdr = fimg.get_header()
		data = fimg.get_data()

		print data.shape

		src = mlab.pipeline.scalar_field(data)
		src.spacing = hdr.get_zooms()

		origin = fimg.get_affine()[:3,3]
		src.origin = origin

		src.update_image_data = True	
		return src


	mlab.figure(bgcolor=(0,0,0), size=(800,600))

	src = getNiftiAsScalarField(args[0])

	mlab.pipeline.iso_surface(src, opacity=0.2, contours=10)

	if options.bg != '':
		mrsrc = getNiftiAsScalarField(options.bg)
		#mlab.pipeline.image_plane_widget(mrsrc, plane_orientation='x_axes',colormap='black-white')
		#mlab.pipeline.image_plane_widget(mrsrc, plane_orientation='y_axes',colormap='black-white' )
		mlab.pipeline.image_plane_widget(mrsrc, plane_orientation='z_axes', colormap='black-white')
		mlab.outline()

	mlab.show()



if __name__ == '__main__':
    parser = OptionParser(usage="Usage: %prog [options] <subject_dir>")
    parser.add_option("-b", "--background", dest="bg", default='', help="Background MRI volume to draw")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        sys.exit(2)
    else:
		run(options, args)
