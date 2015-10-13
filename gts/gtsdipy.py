# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:04:27 2014

@author: dchen
"""
import os
import sys
from os.path import join
import numpy as np
import nibabel as nib
import dipy.io as dio
import dipy.core.gradients as dgrad
from dipy.segment.mask import median_otsu, applymask
from dipy.reconst.csdeconv import auto_response
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

from dipy.data import get_sphere
from dipy.sims.voxel import single_tensor_odf
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.peaks import peaks_from_model
from dipy.align.aniso2iso import resample
from dipy.tracking.eudx import EuDX
import multiprocessing
import dipy.io.pickles as pickle
from tempfile import mkstemp
from dipy.io.dpy import Dpy
import cPickle

def makeIso(filename, outname, voxel_size=1.):
    img = nib.load(filename)
    data = img.get_data()
    zooms = img.get_header().get_zooms()[:3]
    affine = img.get_affine()

    # reslice image into 1x1x1 iso voxel

    new_zooms = (voxel_size, voxel_size, voxel_size)
    data, affine = resample(data, affine, zooms, new_zooms)
    img = nib.Nifti1Image(data, affine)

    print data.shape
    print img.get_header().get_zooms()
    print "###"

    nib.save(img, outname)    

def compCsdPeaks(basename, output, mask=None):
    home = os.getcwd()

    fbase = basename

    fdwi = fbase+".nii.gz"
    fbval = fbase+".bval"
    fbvec = fbase+".bvec"

    print fdwi,fbval,fbvec

    img = nib.load(fdwi)
    data = img.get_data()
    zooms = img.get_header().get_zooms()[:3]
    affine = img.get_affine()

    # reslice image into 1x1x1 iso voxel

#    new_zooms = (1., 1., 1.)
#    data, affine = resample(data, affine, zooms, new_zooms)
#    img = nib.Nifti1Image(data, affine)
#
#    print data.shape
#    print img.get_header().get_zooms()
#    print "###"
#
#    nib.save(img, 'C5_iso.nii.gz')

    bval, bvec = dio.read_bvals_bvecs(fbval, fbvec)
    # invert bvec z for GE scanner
    bvec[:,1]*= -1
    gtab = dgrad.gradient_table(bval, bvec)

    if mask is None:
        print 'generate mask'
        maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50), dilate=2)
    else:
        mask = nib.load(mask).get_data()
        maskdata = applymask(data, mask)



#    tenmodel = dti.TensorModel(gtab)
#    tenfit = tenmodel.fit(data)
#    print('Computing anisotropy measures (FA, MD, RGB)')
#
#
#    FA = fractional_anisotropy(tenfit.evals)
#    FA[np.isnan(FA)] = 0
#
#    fa_img = nib.Nifti1Image(FA.astype(np.float32), img.get_affine())
#    nib.save(fa_img, 'FA.nii.gz')
#
#    return





    # estimate response function, ratio should be ~0.2

    response, ratio = auto_response(gtab, maskdata, roi_radius=10, fa_thr=0.7)
    print response, ratio



    # reconstruct csd model
    print "estimate csd_model"
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    #a_data = maskdata[40:80, 40:80, 60:61]
    #c_data = maskdata[40:80, 59:60, 50:80]
    #s_data = maskdata[59:60, 40:70, 30:80]
    #data_small = a_data
    #
#    evals = response[0]
#    evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    #sphere = get_sphere('symmetric362')
    #csd_fit = csd_model.fit(data_small)
    #csd_odf = csd_fit.odf(sphere)
    #
    #
    #fodf_spheres = fvtk.sphere_funcs(csd_odf, sphere, scale=1, norm=False)
    ##fodf_spheres.GetProperty().SetOpacity(0.4)
    ##
    #fvtk.add(ren, fodf_spheres)
    ##fvtk.add(ren, fodf_peaks)
    #fvtk.show(ren)
    #
    #sys.exit()

    # fit csd peaks
    print "fit csd peaks"
    print "peaks_from_model using core# =" + str(multiprocessing.cpu_count())

    sphere = get_sphere('symmetric724')
    csd_peaks = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 mask=mask,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=True, nbr_processes=10)

    #fodf_peaks = fvtk.peaks(csd_peaks.peak_dirs, csd_peaks.peak_values, scale=1)

#    fd, fname = mkstemp()
#    pickle.save_pickle(fname, csd_peaks)
#
#    os.close(fd)

    #pickle.dump(csd_peaks, open("csd.p", "wb"))


    with open(output, 'wb') as fout:
        cPickle.dump(csd_peaks, fout, -1)


    print "done writing to file %s"% (output)
    return csd_peaks

def loadPeaks(filename):
    with open(filename, 'rb') as fin:
        peaks = cPickle.load(fin)
    return

def loadPeaksFromMrtrix():

    filename='CSD8.nii.gz'
    mask='mask.nii.gz'
    pkdir,pkval,pkind = getPeaksFromMrtrix(filename, mask)
    fodf_peaks = fvtk.peaks(pkdir, pkval, scale=1)
    #fodf_spheres = fvtk.sphere_funcs(data_small, sphere, scale=0.6, norm=False)

    ren = fvtk.ren()
    #fodf_spheres.GetProperty().SetOpacity(0.4)
    fvtk.add(ren, fodf_peaks)
    #fvtk.add(ren, fodf_peaks)
    fvtk.show(ren)

    return fodf_peaks


def runStream(csd_peaks, roi_file, roi_label=1, output_file="tracts.dpy", ang_thr=45., a_low=0.2, step_size=0.1, seeds_per_voxel=30):

    img = nib.load(roi_file)
    roidata = img.get_data()
    p = np.asarray(np.where(roidata == roi_label))
    p = p.transpose()

    seed_points = None
    for i in p:
        points = np.random.uniform(size=[seeds_per_voxel,3]) + (i-0.5)
        if seed_points is None:
            seed_points = points
        else:
            seed_points = np.concatenate([seed_points, points], axis=0)

    sphere = get_sphere('symmetric724')
    print "seed eudx tractography"
    eu = EuDX(csd_peaks.peak_values,
              csd_peaks.peak_indices,
              odf_vertices=sphere.vertices,
              step_sz=step_size,
              seeds=seed_points,
              ang_thr=ang_thr,
              a_low=a_low)

    csa_streamlines_mult_peaks = [streamline for streamline in eu]

    ren = fvtk.ren()

    fvtk.add(ren, fvtk.line(csa_streamlines_mult_peaks, line_colors(csa_streamlines_mult_peaks)))
    fvtk.show(ren)

    dpw = Dpy(output_file, 'w')
    dpw.write_tracks(csa_streamlines_mult_peaks)

    return csa_streamlines_mult_peaks





