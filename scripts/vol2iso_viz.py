#!/usr/bin/python

from optparse import OptionParser
import numpy as np
import nibabel as nib
import sys
from mayavi import mlab
from scipy import interpolate as sinterp

def run(options, args):
	vol2iso_viz(args[0], options.bg, show_label=options.show_label, slice_index=options.slice_index, plane_orientation=options.orientation, save_fig=options.output)

def vol2iso_viz(vol_file, bg_file, bgcolor=(0,0,0), bgslice='', auto_slice=True, show_label=1, force_show_fig=False, save_fig=None, show_outline=False, 
	plane_orientation='z_axes', slice_index=0,
	size=(1024,768),vmin=0.2,nb_labels=10,nb_colors=10,label_orientation='vertical'):
	def getNiftiAsScalarField(filename):
		fimg = nib.load(filename)
		hdr = fimg.get_header()
		data = fimg.get_data()

		print data.shape

		af = fimg.get_affine()
		origin = af[:3,3]
		stride = [af[0,0],af[1,1],af[2,2]]
		print stride

		if stride[0] < 0:
			print 'flipX'
			data = data[::-1,:,:]
			#data = np.flipud(data)
		if stride[1] < 0:
			print 'flipY'			
			data = data[:,::-1,:]
		if stride[2] < 0:
			print 'flipZ'			
			data = data[:,:,::-1]			

		src = mlab.pipeline.scalar_field(data)
		src.spacing = hdr.get_zooms()
		src.origin = origin

		src.update_image_data = True	
		return src, data

	if save_fig:
		mlab.options.offscreen=True
	mlab.figure(bgcolor=bgcolor, size=size)

	# try tp turn on depth peeling
	# https://dl.dropboxusercontent.com/u/1200872/mayavi/depth_peeling.py

	scene = mlab.gcf().scene
	scene.renderer.render_window.set(alpha_bit_planes=1,multi_samples=0)
	scene.renderer.set(use_depth_peeling=True,maximum_number_of_peels=4,occlusion_ratio=0.1)

	# print "Traits for the renderer"
	# scene.renderer.print_traits()
	# print "Traits for the render window"
	# scene.renderer.render_window.print_traits()




	src, data = getNiftiAsScalarField(vol_file)

	iso = mlab.pipeline.iso_surface(src, opacity=0.2, contours=10, vmin=0.2, vmax=1.0)
	iso
	if show_label==1:
		mlab.colorbar(object=iso, nb_labels=10, nb_colors=10, orientation='vertical')


	if plane_orientation=='z_axes':
		mlab.gcf().scene.camera.parallel_projection=True
		mlab.view(azimuth=0, elevation=180, distance='auto', focalpoint='auto')		
	elif plane_orientation=='x_axes':
		mlab.gcf().scene.camera.parallel_projection=True
		mlab.view(azimuth=180, elevation=90, distance='auto', focalpoint='auto')		
	elif plane_orientation=='y_axes':
		mlab.gcf().scene.camera.parallel_projection=True
		mlab.view(azimuth=90, elevation=90, distance='auto', focalpoint='auto')		
	else:
		mlab.gcf().scene.isometric_view()


	if bg_file != '':
		mrsrc, data = getNiftiAsScalarField(bg_file)
		orie = plane_orientation
		if plane_orientation=='iso':
			orie = 'z_axes'

		# from scipy import stats
		# data = stats.threshold(data, threshmin=0.5, threshmax=1, newval=0)			
		# print data.shape

		from scipy.ndimage.measurements import center_of_mass, maximum_position
		com = maximum_position(data)
		print '# max pos = ',com
		if auto_slice:
			if orie=='x_axes':
				slice_index = com[0]
			elif orie=='y_axes':
				slice_index = com[1]			
			elif orie=='z_axes':
				slice_index = com[2]
			else:
				slice_index = com[2]					

		opacity=0.5
		slice_index = int(slice_index)
		# auto flip z-index to below center
		# if plane_orientation=='iso':
		# 	opacity=0.5
		# 	center = data.shape[2]/2
		# 	if slice_index > center:
		# 		d = abs(center-slice_index)
		# 		slice_index = center-d
		# 	slice_index = com[2]

		mlab.pipeline.image_plane_widget(mrsrc, opacity=opacity, plane_orientation=orie, slice_index=int(slice_index), colormap='black-white', line_width=0, reset_zoom=False)

		if show_outline:
			mlab.outline()


	




	if save_fig:
		mlab.savefig(save_fig)
		if force_show_fig:
			mlab.show()
		else:
			mlab.close()

	else:
		mlab.show()		




if __name__ == '__main__':
    parser = OptionParser(usage="Usage: %prog [options] <subject_dir>")
    parser.add_option("-b", "--background", dest="bg", default='', help="Background MRI volume to draw")
    parser.add_option("-o", "--output", dest="output", default=None, help="Figure image to save as output")
    parser.add_option("-l", "--show_label", dest="show_label", default=1, type='int', help="Show label")
    parser.add_option("-s", "--slice_index", dest="slice_index", default=0, help="Slicing index")
    parser.add_option("-r", "--orientation", dest="orientation", default='z_axes', help="Initial view orientation. (x_axes, y_axes, z_axes)")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        sys.exit(2)
    else:
		run(options, args)
