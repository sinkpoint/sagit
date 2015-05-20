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

def compCsdPeaks(basename, output, mask=None, sh_only=False, invert=[1]):
    home = os.getcwd()

    fbase = basename

    fdwi = fbase+".nii.gz"
    fbval = fbase+".bvals"
    fbvec = fbase+".bvecs"

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
    for i in invert:
        bvec[:,i]*= -1
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
    data_small = maskdata
    #
#    evals = response[0]
#    evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    #sphere = get_sphere('symmetric362')

    if sh_only:    
        print 'fitting csd spherical harmonics to data'
        csd_fit = csd_model.fit(data_small)

        outfile = output+'_shfit.dipy'
        with open(outfile, 'wb') as fout:
            cPickle.dump(csd_fit, fout, -1)

        print "done writing to file %s"% (outfile)
        return csd_fit

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
    proc_num = multiprocessing.cpu_count()-1    
    print "peaks_from_model using core# =" + str(proc_num)
    sphere = get_sphere('symmetric724')
    csd_peaks = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 mask=mask,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=True, nbr_processes=proc_num)

    #fodf_peaks = fvtk.peaks(csd_peaks.peak_dirs, csd_peaks.peak_values, scale=1)

#    fd, fname = mkstemp()
#    pickle.save_pickle(fname, csd_peaks)
#
#    os.close(fd)

    #pickle.dump(csd_peaks, open("csd.p", "wb"))

    outfile = output+'_csdpeaks.dipy'
    print 'writing peaks to file...'
    with open(outfile, 'wb') as fout:
        cPickle.dump(csd_peaks, fout, -1)


    print "done writing to file %s"% (outfile)
    return (csd_peaks, outfile)

def loadPeaks(filename):
    with open(filename, 'rb') as fin:
        peaks = cPickle.load(fin)
    return peaks

def loadPeaksFromMrtrix():
    from mrtrix2dipyCSD import getPeaksFromMrtrix
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


def runStream(csd_peaks, roi_file, roi_label=1, ang_thr=45., a_low=0.2, step_size=0.1, seeds_per_voxel=30, out_name=None):

    img = nib.load(roi_file)
    affine = img.get_affine()
    mask_data = img.get_data()
    p = np.asarray(np.where(mask_data == roi_label))
    p = p.transpose()

    # seed_points = None
    # for i in p:
    #     points = np.random.uniform(size=[seeds_per_voxel,3]) + (i-0.5)
    #     if seed_points is None:
    #         seed_points = points
    #     else:
    #         seed_points = np.concatenate([seed_points, points], axis=0)

    import dipy.tracking.utils as utils
    seeds = utils.seeds_from_mask(mask_data==1, density=seeds_per_voxel)    
    print '# of seeds: ',len(seeds)

    sphere = get_sphere('symmetric724')
    print "seed eudx tractography"
    eu = EuDX(csd_peaks.peak_values,
              csd_peaks.peak_indices,
              odf_vertices=sphere.vertices,
              step_sz=step_size,
              seeds=seeds,
              ang_thr=ang_thr,
              a_low=a_low)

    csa_streamlines_mult_peaks = [streamline for streamline in eu]

    out_file = 'tracts.dipy'
    if out_name:
        out_file = out_name+'_'+out_file

        from dipy.io.trackvis import save_trk
        save_trk(out_file, csa_streamlines_mult_peaks, affine,
                 mask.shape)

        dpw = Dpy(out_file, 'w')
        dpw.write_tracks(csa_streamlines_mult_peaks)
        print 'write tracts to %s' % out_file
    return (csa_streamlines_mult_peaks, out_file)




